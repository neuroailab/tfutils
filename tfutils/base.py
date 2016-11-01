from __future__ import absolute_import, division, print_function

import os
import sys
import time
import math
import importlib
import argparse
import json
import copy
import logging
logging.basicConfig()
log = logging.getLogger('tfutils')

import numpy as np
import pymongo
import tensorflow as tf
import gridfs

from tfutils.error import HiLossError
from tfutils.data import CustomQueue
from tfutils.utils import (make_mongo_safe,
                           SONify)

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
                 save=True,
                 restore=True,
                 save_metrics_freq=5,
                 save_valid_freq=3000,
                 cache_filters_freq=3000,
                 cache_path=None,
                 save_filters_freq=30000,
                 tensorboard_dir=None,
                 force_fetch=False,
                 *args, **kwargs):
        """
        Output printer, logger and saver to a database
        Kwargs:
        - save_vars_freq
            0 or None not to save. Otherwise the number of steps.
        - restore_vars_file
            If None, don't save
        NOTE: Not working yet
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

        if self.dosave:
            self.conn = pymongo.MongoClient(host=host, port=port)
            self.collfs = gridfs.GridFS(self.conn[dbname], collname)
            recent_name = '_'.join([dbname, collname, exp_id, '__RECENT'])
            self.collfs_recent = gridfs.GridFS(self.conn[recent_name])

        if tensorboard_dir is not None:  # save graph to tensorboard
            tf.train.SummaryWriter(tensorboard_dir, tf.get_default_graph())

        if cache_path is None:
            self.cache_path = os.path.join(os.environ['HOME'], '.tfutils')
        else:
            self.cache_path = cache_path
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)

        if restore:
            self.restore_model(force_fetch=force_fetch)

        self.start_time_step = time.time()  # start timer

    def restore_model(self, force_fetch=False):
        """
        Fetches record, saves locally, then uses tf's saver.restore
        """
        # fetch record from database and get the filename info from record
        load = self.load_from_db({'exp_id': self.exp_id, 
                                                 'saved_filters': True},
                                 cache_model=True,
                                 force_fetch=force_fetch)
        if load is not None:
            rec, cache_filename = load
            # tensorflow restore
            self.restore(self.sess, cache_filename)
            print('Model variables restored.')
        else:
            init = tf.initialize_all_variables()
            self.sess.run(init)
            print('variables initialized')


    def load_from_db(self, query, cache_model=False, force_fetch=False):
        """
        Loads checkpoint from the database
        Checks the recent and regular checkpoint fs to find the latest one
        matching the query. Returns the GridOut obj corresponding to the
        record.
        Args:
            query: dict of Mongo queries
        """
        count = self.collfs.find(query).count()
        if count > 0:  # get latest that matches query
            ckpt_record = self.collfs.find(query,
                            sort=[('uploadDate', -1)])[0]
            loading_from = 'long-term storage'

        count_recent = self.collfs_recent.find(query).count()
        if count_recent > 0:  # get latest that matches query
            ckpt_record_rec = self.collfs_recent.find(query,
                                sort=[('uploadDate', -1)])[0]
            # use the record with latest timestamp
            if ckpt_record is None or ckpt_record_rec['uploadDate'] > ckpt_record['uploadDate']:
                loading_from = 'recent storage'
                ckpt_record = ckpt_record_rec

        if count + count_recent == 0:  # no matches for query
            log.warning('No matching checkpoint for query "{}"'.format(repr(query)))
            return 

        print('Loading checkpoint from ', loading_from)
        
        if cache_model:
            # should be of form *-1000 (step)
            filename = os.path.basename(ckpt_record.filename)
            cache_filename = os.path.join(self.cache_path, filename)
            # TODO: check that these filenames are unique and correct

            # check if there is no local copy
            if not os.path.isfile(cache_filename) and not force_fetch:
                # create new file to write from gridfs
                load_dest = open(cache_filename, "w+")
                load_dest.close()
                load_dest = open(cache_filename, 'rwb+')
                fsbucket = gridfs.GridFSBucket(ckpt_record._GridOut__files.database,
                                bucket_name=ckpt_record._GridOut__files.name.split('.')[0])
                # save to local disk
                fsbucket.download_to_stream(ckpt_record._id, load_dest)        
        else:
            cache_filename = None            
        #TODO: create numpy-readable format for filter data if user desires
        return ckpt_record, cache_filename

    def save(self, train_res, valid_res):
        elapsed_time_step = time.time() - self.start_time_step
        just_saved = False  # for saving filters

        step = self.global_step.eval(session=self.sess)
        if step % self.save_metrics_freq == 0:
            rec = {'exp_id': self.exp_id,
                   'params': self.SONified_params,
                   'saved_filters': False,
                   'step': step,
                   'duration': int(1000 * elapsed_time_step)}
            if 'optimizer' in train_res:
                del train_res['optimizer']
            rec['train_results'] = train_res
        
            # print loss, learning rate etc
            # TODO: also include error rate of the train set to monitor overfitting
            message = 'Step {} ({:.0f} ms) -- '.format(rec['step'], rec['duration'])
            message += ', '.join(['{}: {:.4f}'.format(k,v) for k,v in train_res.items() if k != 'optimizer'])
            print(message)

            # print validation set performance etc
            if valid_res is not None:
                rec['validation_results'] = valid_res
                message = 'Step {} ({:.0f} ms) validation -- '.format(rec['step'], rec['duration'])
                message += ', '.join('{}: {}'.format(k,v) for k,v in valid_res.items())
                print(message)

            save_rec = SONify(rec)
            make_mongo_safe(save_rec)

            # save filters to db
            save_permanent = step % self.save_filters_freq == 0
            save_cache = self.dosave and step % self.cache_filters_freq == 0
            if save_permanent or save_cache:
                save_rec['saved_filters'] = True 
                saved_path = super(Saver, self).save(self.sess,
                                                     save_path=self.cache_path,
                                                     global_step=step,
                                                     write_meta_graph=False)
                if save_permanent:
                    putfs = self.collfs
                else:
                    putfs = self.collfs_recent
                putfs.put(open(saved_path, 'rb'),
                          filename=saved_path,
                          **save_rec)
            
            if not save_permanent:
                save_rec['saved_filters'] = False
                self.collfs._GridFS__files.insert(save_rec)

            #Vars = tf.all_variables()
            #tmp = int(time.time())
            #for v in Vars:
            #    sdir = '/home/yamins/.tfutils/%d' % tmp
            #    if not os.path.isdir(sdir):
            #        os.makedirs(sdir)
            #    pth = os.path.join(sdir, v.name.replace('/', '__'))
            #    val = v.eval(session=self.sess)
            #    np.save(pth, val)
            
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
    Args:
        - queues (~ data)
        - saver
        - targets
    """

    tf.train.start_queue_runners(sess=sess)
    # start our custom queue runner's threads
    if not hasattr(queues, '__iter__'):
        queues = [queues]
    for queue in queues:
        queue.start_threads(sess)

    # start_time_step = time.time()  # start timer
    #pass_targets = {'loss': train_targets['loss'], 'lr': train_targets['learning_rate']}
    #train_results = sess.run(pass_targets)
    #saver.save(train_results, {})
    print('start training')
    step = global_step.eval(session=sess)
    print("tip", step)
    while step < num_steps:
        # get run output as dictionary {'2': loss2, 'lr': lr, etc..}
        train_results = sess.run(train_targets)
        step = global_step.eval(session=sess)
        if train_results['loss'] > thres_loss:
            raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_results['loss'], thres_loss))

        if step % saver.save_valid_freq  == 0 and valid_targets is not None:
            valid_results = sess.run(valid_targets)
        else:
            valid_results = {}
        # print output, save variables to checkpoint and save loss etc
        saver.save(train_results, valid_results)
    sess.close()


def run_base(model_func,
             model_kwargs,
             train_data_func,
             train_data_kwargs,
             loss_func,
             loss_kwargs,
             learning_rate_func,
             learning_rate_kwargs,
             optimizer_func,
             optimizer_kwargs,
             saver_kwargs,
             train_targets_func=None,
             train_targets_kwargs=None,
             validation=None,
             thres_loss=100,
             seed=None,
             num_steps=1000000,
             log_device_placement=True,
             queue_kwargs=None,
             return_saver=False
             ):
    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      dtype=tf.int64,
                                      trainable=False)

        #train_data_func returns dictionary of iterators, with one key per input to model
        if queue_kwargs is None:
            queue_kwargs = {}
        train_inputs = train_data_func(**train_data_kwargs)
        queue = CustomQueue(train_inputs.node, 
                            train_inputs, 
                            **queue_kwargs)
        train_inputs = queue.batch
        queues = [queue]
        if 'cfg_initial' not in model_kwargs:
            model_kwargs['cfg_initial'] = None
        if 'seed' not in model_kwargs:
            model_kwargs['seed'] = 0
        train_outputs, cfg_final = model_func(inputs=train_inputs, 
                                        train=True, 
                                        **model_kwargs)
        loss = loss_func(train_inputs, train_outputs, **loss_kwargs)
        learning_rate = learning_rate_func(global_step=global_step, **learning_rate_kwargs)
        optimizer_base = optimizer_func(learning_rate=learning_rate, **optimizer_kwargs)
        optimizer = optimizer_base.minimize(loss, global_step)
        train_targets = {'loss': loss, 'learning_rate': learning_rate, 'optimizer': optimizer}
        if train_targets_func is not None:
            if train_targets_kwargs is None:
                train_targets_kwargs = {}
            ttarg = train_targets_func(train_inputs, train_outputs, **train_targets_kwargs)
            train_targets.update(ttarg)

        valid_targetsdict = None
        if validation is not None:
            for vtarg in validation:
                vdatafunc = validation[vtarg]['data_func']
                vdatakwargs = validation[vtarg]['data_kwargs']
                vtargsfunc = validation[vtarg]['targets_func']
                vtargskwargs = validation[vtarg]['targets_kwargs']
                vinputs = vdatafunc(**vdatakwargs)
                new_queue = CustomQueue(vinputs.node, 
                                        vinputs,
                                        **queue_kwargs)
                queues.append(new_queue)
                new_model_kwargs = copy.deepcopy(model_kwargs)
                new_model_kwargs['seed'] = None
                new_model_kwargs['cfg_initial'] = cfg_final
                voutputs, _cfg = model_func(inputs=vinputs, 
                                             train=False, 
                                             **new_model_kwargs)
                assert cfg_final == _cfg, (cfg_final, _cfg)
                vtargets = vtargsfunc(vinputs,
                                      voutputs, 
                                      **vtargskwargs)
                valid_targetsdict[vtarg] = vtargets

        # create session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=log_device_placement))

        model_kwargs_final = copy.deepcopy(model_kwargs)
        model_kwargs_final['cfg_final'] = cfg_final
                  
        params = {'model_func': model_func,
                  'model_kwargs': model_kwargs_final,
                  'train_data_func': train_data_func,
                  'train_data_kwargs': train_data_kwargs,
                  'loss_func': loss_func, 
                  'loss_kwargs': loss_kwargs,
                  'learning_rate_func': learning_rate_func,
                  'learning_rate_kwargs': learning_rate_kwargs,
                  'optimizer_func': optimizer_func,
                  'optimizer_kwargs': optimizer_kwargs,
                  'saver_kwargs': saver_kwargs,
                  'train_targets_func': train_targets_func,
                  'train_targets_kwargs': train_targets_kwargs,
                  'validation': validation,
                  'thres_loss': thres_loss,
                  'seed': seed,
                  'num_steps': num_steps,
                  'log_device_placement': log_device_placement}
        for sk in ['host', 'port', 'dbname', 'collname', 'exp_id']:
            assert sk in saver_kwargs, (sk, saver_kwargs)
        saver = Saver(sess=sess, global_step=global_step, params=params, **saver_kwargs)
        run(sess,
            queues,
            saver,
            train_targets=train_targets,
            global_step=global_step,
            num_steps=num_steps,
            valid_targets=valid_targetsdict,
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
