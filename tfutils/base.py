from __future__ import absolute_import, division, print_function

import os, sys, time, importlib, argparse, json, copy, logging
from collections import OrderedDict

import pymongo
import gridfs
import tensorflow as tf

from tfutils.error import HiLossError
from tfutils.data import CustomQueue
from tfutils.optimizer import ClipOptimizer
import tfutils.utils as utils
from tfutils.utils import make_mongo_safe, SONify

logging.basicConfig()
log = logging.getLogger('tfutils')
log.setLevel('DEBUG')

"""
TODO:
    - There should be a dead-simple way to load a human-readable object (as opposed to being in the
      TF saver binary format) containing filter parameters from a record in the database,
      without having to load up lots of extraneous objects.
    - epoch and batch_num should be added to what is saved.   But how to do that with Queues?
"""


def default_loss_params():
    return {'target': 'labels',
            'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
            'agg_func': tf.reduce_mean}


def default_optimizer_params():
    return {'func': ClipOptimizer,
            'optimizer_class': tf.train.MomentumOptimizer,
            'momentum': 0.9}


class Saver(tf.train.Saver):

    def __init__(self,
                 sess,
                 global_step,
                 params,
                 host,
                 port,
                 dbname,
                 collname,
                 exp_id,
                 restore=True,
                 save=True,
                 save_initial=True,
                 save_metrics_freq=5,
                 save_valid_freq=3000,
                 save_filters_freq=30000,
                 cache_filters_freq=3000,
                 cache_dir=None,
                 tensorboard_dir=None,
                 force_fetch=False,
                 *args, **kwargs):
        """
        :Args:
            - sess (tesorflow.Session)
                Object in which to run calculations
            - global_step (tensorflow.Variable)
                Global step variable, the one that is updated by apply_gradients
            - params (dict)
                Describing all parameters of experiment
            - host (str)
                Hostname where database connection lives
            - port (int)
                Port where database connection lives
            - dbname (str)
                Name of database for storage
            - collname (str)
                Name of collection for storage
            - exp_id (str)
                Experiment id descriptor

        :Kwargs:
            - restore (bool, default: True)
                Whether to restore from saved model
            - save (bool, default: True)
                Whether to save to database
            - save_initial (bool, default: True)
                Whether to save initial model state at step = 0,
            - save_metrics_freq (int, default: 5)
                How often to store train results to database
            - save_valid_freq (int, default: 3000)
                How often to calculate and store validation results to database
            - save_filters_freq (int, default: 30000)
                How often to save filter values to database
            - cache_filters_freq (int, default: 3000)
                How often to cache filter values locally and save to ___RECENT database
            - cache_dir (str, default: None)
                Path where caches will be saved locally. If None, will default to
                ~/.tfutils/<host:post>/<dbname>/<collname>/<exp_id>.
            - tensorboard_dir: (str or None, default: None)
                If not None, directory to put tensorboard stuff.
                If None, tensorboard is disabled
            - force_fetch (bool, default: False)
                Whether to fetch stored model from database even if its locally cached
            - *args, **kwargs
                Additional arguments are passed onto base Saver class constructor
        """

        SONified_params = SONify(params)
        super(Saver, self).__init__(*args, **kwargs)
        self.sess = sess
        self.global_step = global_step
        self.params = params
        self.SONified_params = SONified_params
        self.exp_id = exp_id
        self.dosave = save
        self.save_metrics_freq = save_metrics_freq
        self.save_valid_freq = save_valid_freq
        self.cache_filters_freq = cache_filters_freq
        self.save_filters_freq = save_filters_freq
        self.save_initial = save_initial
        self._restore = restore
        self._restored = False

        if self.dosave:
            self.conn = pymongo.MongoClient(host=host, port=port)
            self.database = self.conn[dbname]
            self.collfs = gridfs.GridFS(self.database, collname)
            self.coll = self.collfs._GridFS__files
            recent_name = '_'.join([dbname, collname, exp_id, '__RECENT'])
            self.collfs_recent = gridfs.GridFS(self.conn[recent_name])
            self.coll_recent = self.collfs_recent._GridFS__files

        if tensorboard_dir is not None:  # save graph to tensorboard
            tf.train.SummaryWriter(tensorboard_dir, tf.get_default_graph())

        if cache_dir is None:
            self.cache_dir = os.path.join(os.environ['HOME'],
                                          '.tfutils',
                                          '%s:%d' % (host, port),
                                          dbname,
                                          collname,
                                          exp_id)
        else:
            self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.load_model(force_fetch=force_fetch)

        self.start_time_step = time.time()  # start timer

    def load_model(self, force_fetch=False):
        """
        Fetches record then uses tf's saver.restore
        """
        # fetch record from database and get the filename info from record
        if self._restore:
            load = self.load_from_db({'exp_id': self.exp_id,
                                      'saved_filters': True},
                                     cache_model=True,
                                     force_fetch=force_fetch)
            if load is not None:
                rec, cache_filename = load
                # tensorflow restore
                self.restore(self.sess, cache_filename)
                self._restored = True
                log.info('Model variables restored from record %s (step %d).'
                         % (str(rec['_id']), rec['step']))

        if not self._restore or load is None:
            init = tf.initialize_all_variables()
            self.sess.run(init)
            log.info('Model variables initialized from scratch.')

    def load_from_db(self, query, cache_model=False, force_fetch=False):
        """
        Loads checkpoint from the database

        Checks the recent and regular checkpoint fs to find the latest one
        matching the query. Returns the GridOut obj corresponding to the
        record.

        Args:
            query: dict expressing MongoDB query
        """
        query['saved_filters'] = True
        count = self.collfs.find(query).count()
        if count > 0:  # get latest that matches query
            ckpt_record = self.coll.find(query,
                                         sort=[('uploadDate', -1)])[0]
            loading_from = self.coll

        count_recent = self.collfs_recent.find(query).count()
        if count_recent > 0:  # get latest that matches query
            ckpt_record_recent = self.coll_recent.find(query,
                                                       sort=[('uploadDate', -1)])[0]
            # use the record with latest timestamp
            if ckpt_record is None or ckpt_record_recent['uploadDate'] > ckpt_record['uploadDate']:
                loading_from = self.coll_recent
                ckpt_record = ckpt_record_recent

        if count + count_recent == 0:  # no matches for query
            log.warning('No matching checkpoint for query "{}"'.format(repr(query)))
            return

        log.info('Loading checkpoint from %s' % loading_from.full_name)

        if cache_model:
            # should be of form *-1000 (step)
            filename = os.path.basename(ckpt_record['filename'])
            cache_filename = os.path.join(self.cache_dir, filename)

            # check if there is no local copy
            if not os.path.isfile(cache_filename) and not force_fetch:
                # create new file to write from gridfs
                load_dest = open(cache_filename, "w+")
                load_dest.close()
                load_dest = open(cache_filename, 'rwb+')
                fsbucket = gridfs.GridFSBucket(self.database,
                                               bucket_name=loading_from.name.split('.')[0])
                fsbucket.download_to_stream(ckpt_record['_id'], load_dest)
        else:
            cache_filename = None
        return ckpt_record, cache_filename

    def save(self, train_res, valid_res):
        """
        Actually saves record into DB and makes local filter caches
        """
        elapsed_time_step = time.time() - self.start_time_step
        duration = 1000 * elapsed_time_step
        just_saved = False  # for saving filters

        step = self.global_step.eval(session=self.sess)

        # TODO: also include error rate of the train set to monitor overfitting
        # DY: I don't understand this TODO -- isn't this already here?
        message = 'Step {} ({:.0f} ms) -- '.format(step, duration)
        msg2 = ['{}: {:.4f}'.format(k,v) for k,v in train_res.items() if k != 'optimizer']
        message += ', '.join(msg2)
        log.info(message)

        save_filters_permanent = step % self.save_filters_freq == 0
        save_filters_tmp = step % self.cache_filters_freq == 0
        save_metrics_now = step % self.save_metrics_freq == 0
        save_valid_now = step % self.save_valid_freq == 0
        need_to_save = self.dosave and (save_filters_permanent or
                        save_filters_tmp or save_metrics_now or save_valid_now)

        rec = {'exp_id': self.exp_id,
               'params': self.SONified_params,
               'saved_filters': False,
               'step': step,
               'duration': duration}
        if 'optimizer' in train_res:
            del train_res['optimizer']
        rec['train_results'] = train_res

        # print validation set performance
        if valid_res:
            rec['validation_results'] = valid_res
            message = 'Step {} validation -- '.format(step)
            message += ', '.join('{}: {}'.format(k,v) for k,v in valid_res.items())
            log.info(message)

        if need_to_save:
            save_rec = SONify(rec)
            make_mongo_safe(save_rec)

            # save filters to db
            if save_filters_permanent or save_filters_tmp:
                save_rec['saved_filters'] = True
                save_path = os.path.join(self.cache_dir, 'checkpoint')
                log.info('Saving model to %s ... ' % save_path)
                saved_path = super(Saver, self).save(self.sess,
                                                     save_path=save_path,
                                                     global_step=step,
                                                     write_meta_graph=False)
                log.info('... done saving.')
                putfs = self.collfs if save_filters_permanent else self.collfs_recent
                log.info('Putting filters into %s database' % repr(putfs))
                with open(saved_path, 'rb') as _fp:
                    putfs.put(_fp, filename=saved_path, **save_rec)
                log.info('... done putting filters into database.')

            if not save_filters_permanent:
                save_rec['saved_filters'] = False
                log.info('Inserting record into database.')
                self.collfs._GridFS__files.insert(save_rec)

        sys.stdout.flush()  # flush the stdout buffer
        self.start_time_step = time.time()


def predict(step, results):
    if not hasattr(results['output'], '__iter__'):
        outputs = [results['outputs']]
    else:
        outputs = results['outputs']

    preds = [tf.argmax(output, 1) for output in outputs]

    return preds


def test(step, results):
    raise NotImplementedError


def run(sess,
        queues,
        saver,
        train_targets,
        global_step,
        num_steps,
        valid_targets=None,
        thres_loss=100):
    """
    Actually runs the evaluation loop.

    :Args:
        - sess: (tesorflow.Session)
            Object in which to run calculations
        - queues (list of CustomQueue)
            Objects containing asynchronously queued data iterators
        - saver (Saver object)
            Saver throughwhich to save results
        - train_targets (dict of tensorflow nodes)
            Targets to train. One item in this dict must be "optimizer" or similar
            to make anything happen
        - num_steps (int)
            How many steps to train to before quitting
        - valid_targets (dict of tensorflow objects, default: None)
            Objects on which validation will be computed
        - thres_loss (float, default: 100)
            If loss exceeds this during training, HiLossError is thrown
    """
    tf.train.start_queue_runners(sess=sess)
    # start our custom queue runner's threads
    if not hasattr(queues, '__iter__'):
        queues = [queues]
    for queue in queues:
        queue.start_threads(sess)

    start_time_step = time.time()  # start timer
    step = global_step.eval(session=sess)
    if step == 0 and saver.save_initial and not saver._restored:
        log.info('Saving initial ...')
        pass_targets = {k:v for k,v in train_targets.items() if k != 'optimizer'}
        train_results = sess.run(pass_targets)
        saver.save(train_results, {})
        log.info('... done saving initial.')

    if step < num_steps:
        log.info('Training beginning ...')
    else:
        log.info('Training cancelled since step (%d) is >= num_steps (%d)' % (step, num_steps))
    while step < num_steps:
        old_step = step
        train_results = sess.run(train_targets)
        step = global_step.eval(session=sess)
        assert (step > old_step), (step, old_step)
        if train_results['loss'] > thres_loss:
            raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_results['loss'], thres_loss))
        if step % saver.save_valid_freq == 0 and valid_targets:
            valid_results = sess.run(valid_targets)
        else:
            valid_results = {}
        saver.save(train_results, valid_results)
    sess.close()


def run_base(saver_params,
             model_params,
             train_params,
             loss_params=None,
             learning_rate_params=None,
             optimizer_params=None,
             validation_params=None,
             thres_loss=100,
             num_steps=1000000,
             log_device_placement=False,
             ):
    """
    Main interface function.

    :Args:
        - saver_params (dict)
            Dictionary of arguments for creating saver object (see Saver class)

        - model_params (dict)
            Containing function that produces model and arguments to that function.
                - model_params['func'] is the function producing the model.
                  The function's signature is:
                    - Must accept:
                        - "inputs" -- data object
                        - "train" -- boolean if training is happening
                        - "cfg_initial" -- dictionary of params to be used to create final config
                        - "seed" -- seed for use in random generation of final config
                    - Must return:
                        - train output tensorflow nodes
                        - final configuration used in model
                - Remaining itmes in model_params are dictionary of arguments massed to func.

        - train_params (dict)
            Containing params for data sources and targets in training:
                - train_params['data'] contains params for the data
                    - train_params['data']['func'] is the function that produces
                      dictionary of data iterators
                    - remainder of train_params['data'] are kwargs passed to func
                - train_params['targets'] (optional) contains params for additional train targets
                    - train_params['targets']['func'] is a function that produces
                      tensorflow nodes as training targets
                    - remainder of train_parms['targets'] are arguments to func
				- train_params['queue_params'] is an optional dict of 
                      params used to specify creation for the queue, passed to the 
                      CustomQueue.__init__ method.   Default is {}. 
    :Kwargs:
        - loss_params (dict):
            Parameters for specifying loss function
                - loss_params['func'] is callable producing the tensorflow node
                  used for training loss
                    - Must accept:
                        - 'inputs'
                        - 'outputs'
                    - Must return:
                        - loss
                - remainder of loss_params are then parameters to func

        - learning_rate_params (dict)
            Parameters for specifying learning_rate:
                - learning_rate_params['func'] is a function producing
                  tensorflow node acting as learning rate.
                  This function must accept argument "global_step".
                - remainder of learning_rate_params are arguments to func.

        - optimizer_params (dict)
            Parameters for creating optimizer:
                - optimizer_params['func'] is a function producing a
                  tensorflow optimizer object (like a subclass of tf.train.Optimizer)
                  - Must accept:
                        "learning_rate" -- the result of the learning_rate_func call
                  - Must return object with a method called "minimize" with
                    the same call signature as tensorflow.train.Optimizer.minimize --- that is:
                        - Must accept:
                            - "loss" -- result of loss_func call
                            - "global_step" -- global step used for determine learning rate,
                        - Must return:
                            - tensorflow node which computes gradients and applies
                              them, and must increment "global_step"
                - Remainder of optimizer_params (aside form "func") are arguments
                  to the optimizer func

        - validation_params (dict)
            Dictionary of validation sources. The structure if this dictionary is:

                {
                    <validation_target_name_1>: {
                        'data': {
                            'func': (callable) data source function for this validation,
                            <kwarg1>: <value1> for 'func',
                            ...
                            },
                        'targets': {
                            'func': (callable) returning targets,
                            <kwarg1>: <value1> for 'func',
                            ...
                            }
						'queue_params': (optional, dict) params for creating queue for
                                this validation. NB: if this is NOT specified, queue params 
                                for this validation default to those used in constructing
                                the training data queue.                             
                    },
                    <validation_target_name_2>: ...
                }

            For each validation_target_name key, the targets are computed and then added to
            the output dictionary to be computed every so often -- unlike train_targets which
            are computed on each time step, these are computed on a basic controlled by the
            valid_save_freq specific in the saver_params.

        - queue_params (dict, defualt: None)
            Dictionary of arguments to CustomQueue object (see
            tfutils.data.CustomQueue documentation)

        - thres_loss (float, default: 100)
            If loss exceeds this during training, HiLossError is thrown

        - num_steps (int, default: 1000000)
            How many total steps of the optimization are run

        - log_device_placement (bool, default: False)
            Whether to log device placement in tensorflow session
    """

    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      dtype=tf.int64,
                                      trainable=False)
        #  train_data_func returns dictionary of iterators, with one key per input to model
        train_data_kwargs = copy.deepcopy(train_params['data'])
        train_data_func = train_data_kwargs.pop('func')    
        train_inputs = train_data_func(**train_data_kwargs)

    	train_queue_params = train_params.get('queue_params', {})
        queue = CustomQueue(train_inputs.node, train_inputs, **train_queue_params)
        queues = [queue]
        train_inputs = queue.batch

        model_kwargs = copy.deepcopy(model_params)
        model_func = model_kwargs.pop('func')
        if 'cfg_initial' not in model_kwargs:
            model_kwargs['cfg_initial'] = None
        if 'seed' not in model_kwargs:
            model_kwargs['seed'] = 0
        train_outputs, cfg_final = model_func(inputs=train_inputs,
                                              train=True,
                                              **model_kwargs)

        if loss_params is None:
            loss_params = default_loss_params()
        loss_kwargs = copy.deepcopy(loss_params)
        loss_func = loss_kwargs.pop('func', utils.get_loss)
        loss = loss_func(train_inputs, train_outputs, **loss_kwargs)

        if learning_rate_params is None:
            learning_rate_params = {}
        learning_rate_kwargs = copy.deepcopy(learning_rate_params)
        learning_rate_func = learning_rate_kwargs.pop('func',
                                                      tf.train.exponential_decay)
        learning_rate = learning_rate_func(global_step=global_step,
                                           **learning_rate_kwargs)

        if optimizer_params is None:
            optimizer_params = default_optimizer_params()
        optimizer_kwargs = copy.deepcopy(optimizer_params)
        optimizer_func = optimizer_kwargs.pop('func', ClipOptimizer)
        optimizer_base = optimizer_func(learning_rate=learning_rate,
                                        **optimizer_kwargs)
        optimizer = optimizer_base.minimize(loss, global_step)

        train_targets = {'loss': loss,
                         'learning_rate': learning_rate,
                         'optimizer': optimizer}
        if train_params.get('targets') is not None:
            ttargs_kwargs = copy.deepcopy(train_params['targets'])
            ttargs_func = ttargs_kwargs.pop('func')
            ttarg = ttargs_func(train_inputs, train_outputs, **ttargs_kwargs)
            train_targets.update(ttarg)

        valid_targets_dict = OrderedDict()
        if validation_params is not None:
            for vtarg in validation_params:
                vdata_kwargs = copy.deepcopy(validation_params[vtarg]['data'])
                vdata_func = vdata_kwargs.pop('func')
                vtargs_kwargs = copy.deepcopy(validation_params[vtarg]['targets'])
                vtargs_func = vtargs_kwargs.pop('func')
                vinputs = vdata_func(**vdata_kwargs)
                
                vqueue_params = validation_params[vtarg].get('queue_params', None)
                if vqueue_params is None:
                	vqueue_params = train_queue_params
                queue = CustomQueue(vinputs.node, vinputs, **vqueue_params)
                queues.append(queue)
                vinputs = queue.batch

                new_model_kwargs = copy.deepcopy(model_kwargs)
                new_model_kwargs['seed'] = None
                new_model_kwargs['cfg_initial'] = cfg_final
                voutputs, _cfg = model_func(inputs=vinputs,
                                            train=False,
                                            **new_model_kwargs)
                assert cfg_final == _cfg, (cfg_final, _cfg)
                vtargets = vtargs_func(vinputs, voutputs, **vtargs_kwargs)
                valid_targets_dict[vtarg] = vtargets

        # create session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=log_device_placement))

        model_kwargs_final = copy.deepcopy(model_kwargs)
        model_kwargs_final['cfg_final'] = cfg_final

        params = {'saver_params': saver_params,
                  'train_params': train_params,
                  'model_params': model_params,
                  'loss_params': loss_params,
                  'learning_rate_params': learning_rate_params,
                  'optimizer_params': optimizer_params,
                  'validation_params': validation_params,
                  'thres_loss': thres_loss,
                  'num_steps': num_steps,
                  'log_device_placement': log_device_placement}
        for sk in ['host', 'port', 'dbname', 'collname', 'exp_id']:
            assert sk in saver_params, (sk, saver_params)
        saver = Saver(sess=sess, global_step=global_step, params=params, **saver_params)
        run(sess,
            queues,
            saver,
            train_targets=train_targets,
            global_step=global_step,
            num_steps=num_steps,
            valid_targets=valid_targets_dict,
            thres_loss=thres_loss)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=json.loads, default=None)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    for p in filter(lambda x: x.endswith('_func'), args):
        modname, objname = args[p].rsplit('.', 1)
        mod = importlib.import_module(modname)
        args[p] = getattr(mod, objname)
    return args


def main():
    args = get_params()
    run_base(**args)


"""
Something like this could be used to create and save variables
in a readable format.
    def save_variables_to_readable_format()
        Vars = tf.all_variables()
        tmp = int(time.time())
        for v in Vars:
        sdir = '/home/yamins/.tfutils/%d' % tmp
        if not os.path.isdir(sdir):
            os.makedirs(sdir)
            pth = os.path.join(sdir, v.name.replace('/', '__'))
            val = v.eval(session=self.sess)
            np.save(pth, val)

"""
