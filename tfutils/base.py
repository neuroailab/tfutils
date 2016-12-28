from __future__ import absolute_import, division, print_function

import os, sys, time, importlib, argparse, json, copy, logging
from collections import OrderedDict

import pymongo
import gridfs
import tensorflow as tf

from tfutils.error import HiLossError, NoGlobalStepError, NoChangeError
from tfutils.data import Queue
from tfutils.optimizer import ClipOptimizer
import tfutils.utils as utils
from tfutils.utils import make_mongo_safe, sonify

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


DEFAULT_SAVE_PARAMS = {'save_metrics_freq': 100,
                       'save_valid_freq': 3000,
                       'cache_filters_freq': 3000,
                       'save_filters_freq': 30000,
                       'save_initial': True,
                       'save_to_gfs': (),
                       'do_save': True}


DEFAULT_LOAD_PARAMS = {'do_restore': True}


class DBInterface(object):

    def __init__(self,
                 host=None,
                 port=None,
                 dbname=,
                 collname=,
                 exp_id=,
                 cache_dir=None
                 ):

        self.host = host
        self.port = port
        self.dbname = dbname
        self.collname = collname
        self.exp_id = exp_id

        self.conn = pymongo.MongoClient(host=self.host, port=self.port)

        self.collfs = gridfs.GridFS(self.conn[self.dbname], self.collname)
        recent_name = '_'.join([self.dbname, self.collname, self.exp_id, '__RECENT'])
        self.collfs_recent = gridfs.GridFS(self.conn[recent_name])

        self.coll = self.collfs._GridFS__files
        self.coll_recent = self.collfs_recent._GridFS__files

        # setup cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(os.environ['HOME'],
                                          '.tfutils',
                                          '{}-{}'.format(self.host, self.port),
                                          self.dbname,
                                          self.collname,
                                          self.exp_id)
        else:
            self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
            # TODO: if do_save false or save_filters_freq == 0, then don't create

    def get_record(self, query):
        """
        Loads checkpoint from the database

        Checks the recent and regular checkpoint fs to find the latest one
        matching the query. Returns the GridOut obj corresponding to the
        record.

        Args:
            query: dict expressing MongoDB query
        """
        # get latest from permanent DB that matches query
        try:
            ckpt_record = self.coll.find(query, sort=[('uploadDate', -1)])[0]
        except:
            ckpt_record = None

        # get latest from recent DB that matches query
        try:
            ckpt_record_recent = self.coll_recent.find(query, sort=[('uploadDate', -1)])[0]
        except:
            ckpt_record_recent = None

        if ckpt_record is None and ckpt_record_recent is None:
            log.warning('No matching checkpoint for query "{}"'.format(repr(query)))
        elif ckpt_record is None or ckpt_record_recent['uploadDate'] > ckpt_record['uploadDate']:
                log.info('Loading checkpoint from %s' % self.coll_recent.full_name)
                ckpt_record = ckpt_record_recent
        else:
            log.info('Loading checkpoint from %s' % self.coll.full_name)

        return ckpt_record


class SaveToDB(DBInterface):

    def __init__(self,
                 sess=None,
                 global_step=None,
                 **dbparams,
                 *tfsaver_args,
                 **tfsaver_kwargs
                 ):
        super(SaveToDB, self).__init__(**dbparams)
        self.sess = sess
        self.global_step = global_step
        self.sonified_params = utils.sonify(params)

        if save_params is None:
            save_params = {}
        self.save_params = copy.deepcopy(DEFAULT_SAVE_PARAMS)
        self.save_params.update(save_params)

        self.tf_saver = tf.train.Saver(*tfsaver_args, **tfsaver_kwargs)

    def save(self, train_res=None, valid_res=None):
        """
        Actually saves record into DB and makes local filter caches
        """
        elapsed_time_step = time.time() - self.start_time_step
        duration = 1000 * elapsed_time_step
        need_to_save = False

        rec = {'exp_id': self.exp_id,
                'params': self.sonified_params,
                'saved_filters': False,
                'duration': duration}

        save_initial = step == 0 and dbinterface.do_save and dbinterface.save_initial and not dbinterface.load_data

        if train_res is not None:
            if not hasattr(self.global_step, 'eval'):
                raise NoGlobalStepError('When training, you must pass global_step'
                                        ' tensorflow operation to the saver.')
            step = self.global_step.eval(session=self.sess)
            rec['step'] = step
            message = 'Step {} ({:.0f} ms) -- '.format(step, duration)
            msg2 = ['{}: {:.4f}'.format(k,v) for k,v in train_res.items() if k != 'optimizer']
            message += ', '.join(msg2)
            log.info(message)

            save_filters_permanent = step % self.save_filters_freq == 0
            save_filters_tmp = step % self.cache_filters_freq == 0
            save_metrics_now = step % self.save_metrics_freq == 0
            save_valid_now = step % self.save_valid_freq == 0
            need_to_save = self.do_save and (save_filters_permanent or
                            save_filters_tmp or save_metrics_now or save_valid_now)

            if 'optimizer' in train_res:
                del train_res['optimizer']
            rec['train_results'] = train_res
        else:
            save_filters_permanent = save_filters_tmp = False
            need_to_save = True
            rec['validation_only'] = True
            rec['validates'] = self.load_data[0]['_id']

        # print validation set performance
        if valid_res is not None:
            rec['validation_results'] = valid_res
            message = 'Validation -- '
            message += ', '.join('{}: {}'.format(k,v) for k,v in valid_res.items() if not k in self.save_to_gfs)
            log.info(message)

        if need_to_save:
            save_rec = sonify(rec)
            save_to_gfs = {}
            for _k in self.save_to_gfs:
                save_to_gfs[_k] = save_rec.pop(_k)
            make_mongo_safe(save_rec)

            # save filters to db
            if save_filters_permanent or save_filters_tmp:
                save_rec['saved_filters'] = True
                save_path = os.path.join(self.cache_dir, 'checkpoint')
                log.info('Saving filters to %s ... ' % save_path)
                saved_path = self.tf_saver.save(self.sess,
                                                save_path=save_path,
                                                global_step=step,
                                                write_meta_graph=False)
                log.info('... done saving.')
                putfs = self.collfs if save_filters_permanent else self.collfs_recent
                log.info('Putting filters into %s database' % repr(putfs))
                with open(saved_path, 'rb') as _fp:
                    outrec = putfs.put(_fp, filename=saved_path, **save_rec)
                log.info('... done putting filters into database.')

            if not save_filters_permanent:
                save_rec['saved_filters'] = False
                log.info('Inserting record into database.')
                outrec = self.collfs._GridFS__files.insert_one(save_rec)

            if save_to_gfs:
                idval = str(outrec['_id'])
                save_to_gfs_path = idval + "_fileitems"
                self.collfs.put(json.dumps(save_to_gfs), filename=save_to_gfs_path)

        sys.stdout.flush()  # flush the stdout buffer


# class LoadFromDB(DBInterface):

#     def __init__(self,
#                  sess,
#                  query=None,
#                  **kwargs
#                  ):
#         """
#         :Kwargs:
#             - params (dict)
#                 Describing all parameters of experiment
#             - save_params (dict)
#                 Describing the parameters need to construct the save database, and
#                 control saving.  These include:
#                     - host (str)
#                         Hostname where database connection lives
#                     - port (int)
#                         Port where database connection lives
#                     - dbname (str)
#                         Name of database for storage
#                     - collname (str)
#                         Name of collection for storage
#                     - exp_id (str)
#                         Experiment id descriptor

#                     - do_save (bool, default: True)
#                         Whether to save to database
#                     - save_initial (bool, default: True)
#                         Whether to save initial model state at step = 0,
#                     - save_metrics_freq (int, default: 5)
#                         How often to store train results to database
#                     - save_valid_freq (int, default: 3000)
#                         How often to calculate and store validation results to database
#                     - save_filters_freq (int, default: 30000)
#                         How often to save filter values to database
#                     - cache_filters_freq (int, default: 3000)
#                         How often to cache filter values locally and save
#                         to ___RECENT database
#             - load_params (dict)
#                 Similar to save_params, if you want loading to happen from a different
#                 location than where saving occurs.   Parameters include:
#                     - host (str)
#                         Hostname where database connection lives
#                     - port (int)
#                         Port where database connection lives
#                     - dbname (str)
#                         Name of database for storage
#                     - collname (str)
#                         Name of collection for storage
#                     - exp_id (str)
#                         Experiment id descriptor
#                     - load_query (dict)
#                         mongodb query describing how to load from loading database
#             - do_restore (bool, default: True)
#                 Whether to restore from saved model
#             - sess (tesorflow.Session)
#                 Object in which to run calculations.  This is required if actual loading/
#                 saving is going to be done (as opposed to just e.g. getting elements from
#                 the MongoDB).
#             - global_step (tensorflow.Variable)
#                 Global step variable, the one that is updated by apply_gradients.  This
#                 is required if being using in a training context.
#             - cache_dir (str, default: None)
#                 Path where caches will be saved locally. If None, will default to
#                 ~/.tfutils/<host-port>/<dbname>/<collname>/<exp_id>.
#             - *tfsaver_args, **tsaver_kwargs
#                 Additional arguments to be passed onto base Saver class constructor
#         """

#         super(LoadFromDB, self).__init__(**kwargs)
#         self.sess = sess
#         self.query = query if query is not None else {}


def initialize(host=None,
                 port=None,
                 dbname=,
                 collname=,
                 exp_id=,
                 cache_dir=None,
                 query=None,
                 do_restore=False):
    """
    Fetches record then uses tf's saver.restore

    Possibilities:
    load params:
    - not specified:
        infer same as save params

    - do_restore == True:
        - same as save params:
            load from load_db:
            - entry exists in db:
                get it
                - global step == last step:
                ok (continue training)
                - global step != last step:
                Error
            - entry is not in db:
                Error
        - different from save params:
            load from load_db:
                - entry exists in db:
                    get it
                - entry is not in db:
                    Error
    - do_restore == False:
        load_params: ignore
        init from scratch
    """
    db = DBInterface(host=host, port=port, dbname=dbname, collname=collname,
                exp_id=exp_id, cache_dir=cache_dir)
    query = query if query is not None else {}

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=log_device_placement))


    # fetch record from database and get the filename info from record
    if do_restore:
        query['saved_filters'] = True
        rec = db.get_record(query)
        if rec is None:
            raise Exception('You specified load parameters but no record '
                            'was found with the given spec.')
        # save filters locally
        # should be of form *-1000 (step)
        filename = os.path.basename(ckpt_record['filename'])
        cache_filename = os.path.join(cache_dir, filename)

        # check if there is no local copy
        if not os.path.isfile(cache_filename):
            log.info('No cache file at %s, loading from DB' % cache_filename)
            # create new file to write from gridfs
            load_dest = open(cache_filename, "w+")
            load_dest.close()
            load_dest = open(cache_filename, 'rwb+')
            fsbucket = gridfs.GridFSBucket(loading_from._Collection__database,
                                bucket_name=loading_from.name.split('.')[0])
            fsbucket.download_to_stream(ckpt_record['_id'], load_dest)
        else:
            log.info('Cache file found at %s, using that to load' % cache_filename)

        # tensorflow restore
        log.info('Restoring variables from record {} (step {})...'.format(rec['_id'], rec['step']))
        tf_saver.restore(sess, cache_filename)
        log.info('... done restoring.')

    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        log.info('Model variables initialized from scratch.')




    # count = collfs.find(query).count()
    # if count > 0:
    #     ckpt_record = coll.find(query, sort=[('uploadDate', -1)])[0]
    #     loading_from = coll
    # else:
    #     ckpt_record = None

    # # get latest from temporary DB that matches query
    # count_recent = collfs_recent.find(query).count()
    # if count_recent > 0:
    #     ckpt_record_recent = coll_recent.find(query, sort=[('uploadDate', -1)])[0]
    #     # use the record with latest timestamp
    #     if ckpt_record is None or ckpt_record_recent['uploadDate'] > ckpt_record['uploadDate']:
    #         loading_from = coll_recent
    #         ckpt_record = ckpt_record_recent

    # if count == 0 and count_recent == 0:  # no matches for query
    #     log.warning('No matching checkpoint for query "{}"'.format(repr(query)))
    #     return

    # log.info('Loading checkpoint from %s' % loading_from.full_name)

    # return ckpt_record




def predict(step, results):
    if not hasattr(results['output'], '__iter__'):
        outputs = [results['outputs']]
    else:
        outputs = results['outputs']

    preds = [tf.argmax(output, 1) for output in outputs]

    return preds


def get_valid_results(sess, valid_targets):
    """
    Helper function for actually computing validation results.
    """
    valid_results = {}
    for targname in valid_targets:
        num_steps = valid_targets[targname]['num_steps']
        targ = valid_targets[targname]['targets']
        agg_func = valid_targets[targname]['agg_func']
        ress = []
        for _step in range(num_steps):
            res = sess.run(targ)
            ress.append(res)
        if agg_func:
            valid_results[targname] = agg_func(ress)
        else:
            valid_results[targname] = ress
    return valid_results


def start_queues(sess, queues):
    """Helper function for starting queues before running processes.
    """
    tf.train.start_queue_runners(sess=sess)
    # start our queue runner's threads
    if not hasattr(queues, '__iter__'):
        queues = [queues]
    for queue in queues:
        queue.start_threads(sess)


def stop_queues(sess, queues):
    """Helper function for stopping queues cleanly.
    """
    if not hasattr(queues, '__iter__'):
        queues = [queues]
    for queue in queues:
        queue.stop_threads(sess)


def get_var_names():
    return [x.name for x in tf.global_variables()]


def get_op_names():
    return [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]


def _get_features(inputs, outputs, layer):
    return {'features': tf.get_default_graph().get_tensor_by_name(layer)}


def get_features(data, name, **params):



    default_val = {
        'data': {'func': data,
                 'batch_size': 1},
        'queue_params': {'queue_type': 'fifo',
                         'batch_size': 1,
                         'n_threads': 1},
        'num_steps': 1
    }

    params['validation_params']['']['targets']['func'] = _get_features
    return test_from_params(**params)


def test(sess,
         queues,
         dbinterface,
         valid_targets):

    """
    Actually runs the testing evaluation loop.

    :Args:
        - sess: (tesorflow.Session)
            Object in which to run calculations
        - queues (list of CustomQueue)
            Objects containing asynchronously queued data iterators
        - dbinterface (DBInterface object)
            Saver through which to save results
        - valid_targets (dict of tensorflow objects)
            Objects on which validation will be computed
    """
    start_queues(sess, queues)
    dbinterface.start_time_step = time.time()
    valid_results = get_valid_results(sess, valid_targets)
    dbinterface.save({}, valid_results)
    stop_queues(sess, queues)
    sess.close()
    return valid_results


def test_from_params(load_params,
              model_params,
              validation_params,
              log_device_placement=False,
              save_params=None,
              **kwargs):

    """
    Main testing interface function.

    Same as train_from_parameters; but just performs testing without training.

    For documentation, see argument descriptions in train_from_params.
    """


    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]

        # create session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=log_device_placement))

        model_kwargs = copy.deepcopy(model_params)
        model_func = model_kwargs.pop('func')
        dbinterface = DBInterface(load_params=load_params)
        dbinterface.load_rec()
        cfg_final = dbinterface.load_data[0]['params']['model_params']['cfg_final']
        original_seed = dbinterface.load_data[0]['params']['model_params']['seed']
        train_queue_params = dbinterface.load_data[0]['params']['train_params'].get('queue_params',
                                                                                    {})
        valid_targets_dict, queues = get_valid_targets_dict(validation_params,
                                                            model_func, model_kwargs,
                                                            train_queue_params,
                                                            cfg_final,
                                                            original_seed)
        model_params['cfg_final'] = cfg_final
        load_params['do_restore'] = True
        params = {'load_params': load_params,
                  'sav_params': save_params,
                  'model_params': model_params,
                  'validation_params': validation_params,
                  'log_device_placement': log_device_placement}
        dbinterface = DBInterface(sess=sess, params=params,
                                  load_params=load_params, save_params=save_params)
        dbinterface.initialize()
        return test(sess,
                    queues,
                    dbinterface,
                    valid_targets=valid_targets_dict)





def train_from_params(save_params,
               model_params,
               train_params,
               loss_params=None,
               learning_rate_params=None,
               optimizer_params=None,
               validation_params=None,
               log_device_placement=False,
               load_params=None
             ):
    """
    Main training interface function.

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
                      Queue.__init__ method.   Default is {}.
                - thres_loss (float, default: 100)
                    If loss exceeds this during training, HiLossError is thrown
                - num_steps (int, default: 1000000)
                    How many total steps of the optimization are run
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
            Dictionary of arguments to Queue object (see
            tfutils.data.Queue documentation)



      - load_params (dict)
            Dictionary of arguments for loading model, if different from saver
            (see Saver class)

        - log_device_placement (bool, default: False)
            Whether to log device placement in tensorflow session
    """

    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      dtype=tf.int64,
                                      trainable=False)

        queue, train_params['data'] = call(data,
                                            train_params['data'],
                                            queue_params=queue_params)
        train_inputs = queue.batch
        queues = [queue]
        train_outputs, model_params = call(model,
                                           model_params,
                                           train_inputs,
                                           train=True)
        loss_out, loss_params = call(loss,
                                     loss_params,
                                     train_inputs,
                                     train_outputs)
        learning_rate_out, learning_params = call(learning_rate,
                                                  learning_rate_params,
                                                  global_step)
        optimizer_out, optimizer_params = call(optimizer,
                                               optimizer_params,
                                               learning_rate_out,
                                               loss_out,
                                               global_step)

        train_targets = {'loss': loss_out,
                         'learning_rate': learning_rate_out,
                         'optimizer': optimizer_out}

        if train_params.get('targets') is not None:
            t, train_params['targets'] = call(target,
                                              train_params['targets'],
                                              train_inputs,
                                              train_outputs)
            # TODO: assumes that target func returns a dict;
            # is that a safe assumption or should it be somehow enforced?
            train_targets.update(t)

        # validation
        scope = tf.get_variable_scope()
        scope.reuse_variables()

        if validation_params is None:
            validation_params = {}

        validation_targets = {}
        for target_name, val_params in validation_params.items():
            with tf.name_scope('validation/{}'.format(target_name)):
                val_queue, val_params['data'] = call(data,
                                            val_params['data'],
                                            queue_params=val_params['queue_params'])
                val_inputs = val_queue.batch
                queues.extend(val_queue)
                val_outputs, _ = call(model,
                                      model_params,
                                      val_inputs,
                                      train=False)
                if val_params.get('targets') is not None:
                    t, val_params['targets'] = call(target,
                                                    train_params['targets'],
                                                    train_inputs,
                                                    train_outputs)
                    # TODO: assumes that target func returns a dict;
                    # is that a safe assumption or should it be somehow enforced?
                    validation_targets.update(t)

        # create session
        
        model_params['cfg_final'] = cfg_final
        params = {'save_params': save_params,
                  'load_params': load_params,
                  'train_params': train_params,
                  'model_params': model_params,
                  'loss_params': loss_params,
                  'learning_rate_params': learning_rate_params,
                  'optimizer_params': optimizer_params,
                  'validation_params': validation_params,
                  'log_device_placement': log_device_placement}

        with initialize(**load_params) as sess:
            saver = SaveToDB(sess=sess, params=saver_params, global_step=global_step)

            train(sess,
                queues,
                saver=saver,
                train_targets=train_targets,
                valid_targets=valid_targets_dict,
                global_step=global_step,
                num_steps=train_params['num_steps'],
                thres_loss=train_params['thres_loss'])


def train(sess,
        queues,
        saver=None,
        train_targets,
        global_step,
        num_steps,
        valid_targets=None,
        thres_loss=100):
    """
    Actually runs the training evaluation loop.

    :Args:
        - sess: (tesorflow.Session)
            Object in which to run calculations
        - queues (list of Queue)
            Objects containing asynchronously queued data iterators
        - dbinterface (DBInterface object)
            Saver through which to save results
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
    if not hasattr(queues, '__iter__'):
        queues = [queues]

    tf.train.start_queue_runners(sess=sess)
    for queue in queues:
        queue.start_threads(sess)

    step = global_step.eval(session=sess)

    if step < num_steps:
        log.info('Training beginning ...')
    else:
        log.info('Training cancelled since step (%d) is >= num_steps (%d)' % (step, num_steps))

    while step < num_steps:
        old_step = step
        saver.start_time_step = time.time()
        train_results = sess.run(train_targets)
        step = global_step.eval(session=sess)

        if step <= old_step:
            raise NoChangeError('Your optimizer should have incremented the global step, '
                                'but did not: old_step=%d, new_step=%d' % (old_step, step))

        if train_results['loss'] > thres_loss:
            raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(
                                        train_results['loss'], thres_loss))

        if step % saver.save_valid_freq == 0 and valid_targets is not None:
            valid_results = get_valid_results(sess, valid_targets)
        else:
            valid_results = {}

        saver.save(train_res=train_results, valid_res=valid_results)

    log.info('Done training.')

    for queue in queues:
        queue.stop_threads(sess)



def get_valid_targets_dict(validation_params,
                           model_func, model_kwargs,
                           default_queue_params,
                           cfg_final,
                           original_seed):
    """Helper function for creating validation target operations.
       NB: this function may modify validation_params"""
    valid_targets_dict = OrderedDict()
    queues = []
    for vtarg in validation_params:
        if 'targets' not in validation_params[vtarg]:
            validation_params[vtarg]['targets'] = default_loss_params()
        if 'func' not in validation_params[vtarg]['targets']:
            validation_params[vtarg]['targets']['func'] = utils.get_loss
        if 'agg_func' not in validation_params[vtarg]:
            validation_params[vtarg]['agg_func'] = None
        if 'queue_params' not in validation_params[vtarg] and default_queue_params:
            validation_params[vtarg]['queue_params'] = default_queue_params

        vdata_kwargs = copy.deepcopy(validation_params[vtarg]['data'])
        vdata_func = vdata_kwargs.pop('func')
        vtargs_kwargs = copy.deepcopy(validation_params[vtarg]['targets'])
        vtargs_func = vtargs_kwargs.pop('func')
        agg_func = validation_params[vtarg]['agg_func']
        vinputs = vdata_func(**vdata_kwargs)
        if 'num_steps' not in validation_params[vtarg]:
            validation_params[vtarg]['num_steps'] = vinputs.total_batches
        num_steps = validation_params[vtarg]['num_steps']
        vqueue_params = validation_params[vtarg].get('queue_params', {})
        queue = Queue(vinputs.node, vinputs, **vqueue_params)
        queues.append(queue)
        vinputs = queue.batch
        new_model_kwargs = copy.deepcopy(model_kwargs)
        new_model_kwargs['seed'] = original_seed
        new_model_kwargs['cfg_initial'] = cfg_final
        with tf.name_scope('validation/%s' % vtarg):
            voutputs, _cfg = model_func(inputs=vinputs,
                                    train=True,
                                    **new_model_kwargs)
            tf.get_variable_scope().reuse_variables()
            vtargets = vtargs_func(vinputs, voutputs, **vtargs_kwargs)
            valid_targets_dict[vtarg] = {'targets': vtargets,
                                         'agg_func': agg_func,
                                         'num_steps': num_steps}
    return valid_targets_dict, queues


def call(base_func, params, *args, **kwargs):
    params_copy = copy.deepcopy(params)
    func = params_copy['func']
    return base_func(func=func, *args, **params_copy, **kwargs)


def data(func, queue_params=None, **params):
    if queue_params is None:
        queue_params = {}
    inputs = func(**params)
    queue = Queue(inputs, **queue_params)
    return queue, params


def model(func, inputs, train=False, seed=0, **params):
    outputs, params_final = func(inputs, train=train, seed=seed, **params)
    params_final['train'] = train
    params_final['seed'] = seed
    return outputs, params_final


def loss(inputs, outputs,
         func=utils.get_loss,
         target='labels',
         loss_per_case_func=tf.nn.sparse_softmax_cross_entropy_with_logits,
         agg_func=tf.reduce_mean,
         **params):
    outputs = func(inputs, outputs, target=target,
                   loss_per_case_func=loss_per_case_func,
                   agg_func=agg_func, **params)
    params_final = copy.deepcopy(params)
    params_final['func'] = func
    params_final['target'] = target
    params_final['loss_per_case_func'] = loss_per_case_func
    params_final['agg_func'] = agg_func
    return outputs, params_final


def learning_rate(global_step,
                  func=tf.train.exponential_decay,
                  **params):
    outputs = func(global_step=global_step, **params)
    params_final = copy.deepcopy(params)
    params_final['func'] = func
    return outputs, params_final


def optimizer(learning_rate, loss, global_step,
              func=ClipOptimizer,
              optimizer_class=tf.train.MomentumOptimizer,
              **params):
    if optimizer_class == tf.train.MomentumOptimizer:
        params['momentum'] = .9
    opt = func(optimizer_class=optimizer_class, learning_rate=learning_rate,
               global_step=global_step, **params)
    outputs = opt.minimize(loss, global_step)
    params_final = copy.deepcopy(params)
    params_final['func'] = func
    params_final['optimizer_class'] = optimizer_class
    return outputs, params_final    


def train_target(func, inputs, outputs, **params):
    return func(inputs, outputs, **params), params


def validation_target(inputs, output,
                      func=utils.get_loss,
                      agg_func=None,
                      num_steps=np.inf,
                      **params):
    outputs = func(inputs, outputs, agg_func=agg_func, **params)
    params_final = copy.deepcopy(params)
    params_final['func'] = func
    params_final['agg_func'] = agg_func
    return outputs, params_final


def validation(params):
    params2obj(loss, params['targets'])#, agg_func=None)
    with tf.name_scope('validation/%s' % vtarg):
        if 'agg_func' not in validation_params[vtarg]:
            validation_params[vtarg]['agg_func'] = None
        if 'num_steps' not in validation_params[vtarg]:
            validation_params[vtarg]['num_steps'] = vinputs.total_batches

        val_inputs = params2obj(data, val_params['data'], queue_params=params['queue_params'])
        val_outputs, cfg_final = params2obj(model, model_params, inputs=val_inputs, train=False)
        loss_out = params2obj(loss, loss_params, inputs=val_inputs, outputs=val_outputs)

        val_outputs, _cfg = params2obj(model, inputs=val_inputs, train=False)
        tf.get_variable_scope().reuse_variables()
        params['targets'].update(agg_func=None, )
        val_targets = params2obj(target, params['targets'], val_inputs, val_outputs)
        val_targets = {'targets': val_targets,
                    'agg_func': agg_func,
                    'num_steps': num_steps}

    return val_targets


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
    train_from_params(**args)


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
