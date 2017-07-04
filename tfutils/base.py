from __future__ import absolute_import, division, print_function

import os
import sys
import time
import importlib
import argparse
import json
import copy
import logging
import tarfile
import cPickle
import threading
from collections import OrderedDict

import tqdm
import pymongo
from pymongo import errors as er
from bson.objectid import ObjectId
import gridfs
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.ops import variables
import numpy as np

from tfutils.error import HiLossError, NoGlobalStepError, NoChangeError
from tfutils.data import get_queue
from tfutils.optimizer import ClipOptimizer
import tfutils.utils as utils
from tfutils.utils import (make_mongo_safe,
                           sonify,
                           get_saver_pb2_v2_files,
                           verify_pb2_v2_files,
                           frozendict)

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

if 'TFUTILS_HOME' in os.environ:
    TFUTILS_HOME = os.environ['TFUTILS_HOME']
else:
    TFUTILS_HOME = os.path.join(os.environ['HOME'], '.tfutils')

DEFAULT_DEVICES = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

DEFAULT_LOOP_PARAMS = frozendict({})

DEFAULT_LOSS_PARAMS = frozendict({'targets': ['labels'],
                                  'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
                                  'agg_func': tf.reduce_mean})

DEFAULT_OPTIMIZER_PARAMS = frozendict({'optimizer_class': tf.train.MomentumOptimizer,
                                       'momentum': 0.9})

DEFAULT_TRAIN_NUM_STEPS = None

DEFAULT_TRAIN_THRES_LOSS = 100

DEFAULT_SAVE_PARAMS = frozendict({'save_metrics_freq': 100,
                                  'save_valid_freq': 3000,
                                  'cache_filters_freq': 3000,
                                  'save_filters_freq': 30000,
                                  'save_initial_filters': True,
                                  'save_to_gfs': (),
                                  'do_save': True})


DEFAULT_LEARNING_RATE_PARAMS = frozendict({'func': tf.train.exponential_decay})

DEFAULT_LOAD_PARAMS = frozendict({'do_restore': True})

DEFAULT_PARAMS = frozendict({
    'dont_run': False,
    'model_params': {},
    'train_params': {},
    'validation_params': {},
    'log_device_placement': False,
    'inter_op_parallelism_threads': 40,
    'save_params': dict(DEFAULT_SAVE_PARAMS),
    'load_params': dict(DEFAULT_LOAD_PARAMS),
    'loss_params': dict(DEFAULT_LOSS_PARAMS),
    'optimizer_params': dict(DEFAULT_OPTIMIZER_PARAMS),
    'learning_rate_params': dict(DEFAULT_LEARNING_RATE_PARAMS),
})


class DBInterface(object):

    def __init__(self,
                 params=None,
                 save_params=None,
                 load_params=None,
                 sess=None,
                 global_step=None,
                 cache_dir=None,
                 *tfsaver_args,
                 **tfsaver_kwargs):
        """
        :Kwargs:
            - params (dict)
                Describing all parameters of experiment
            - save_params (dict)
                Describing the parameters need to construct the save database, and
                control saving.  These include:
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
                        NOTE: the variables host/port/dbname/coll/exp_id control
                        the location of the saved data for the run, in order of
                        increasing specificity.  When choosing these, note that:
                            1.  If a given host/port/dbname/coll/exp_id already has saved checkpoints,
                                then any new call to start training with these same location variables
                                will start to train from the most recent saved checkpoint.  If you mistakenly
                                try to start training a new model with different variable names, or structure,
                                from that existing checkpoint, an error will be raised, as the model will be
                                incompatiable with the saved variables.
                            2.  When choosing what dbname, coll, and exp_id, to use, keep in mind that mongodb
                                queries only operate over a single collection.  So if you want to analyze
                                results from a bunch of experiments together using mongod queries, you should
                                put them all in the same collection, but with different exp_ids.  If, on the
                                other hand, you never expect to analyze data from two experiments together,
                                you can put them in different collections or different databases.  Choosing
                                between putting two experiments in two collections in the same database
                                or in two totally different databases will depend on how you want to organize
                                your results and is really a matter of preference.
                    - do_save (bool, default: True)
                        Whether to save to database
                    - save_initial_filters (bool, default: True)
                        Whether to save initial model filters at step = 0,
                    - save_metrics_freq (int, default: 5)
                        How often to store train results to database
                    - save_valid_freq (int, default: 3000)
                        How often to calculate and store validation results
                                                to database
                    - save_filters_freq (int, default: 30000)
                        How often to save filter values to database
                    - cache_filters_freq (int, default: 3000)
                        How often to cache filter values locally and save
                        to ___RECENT database
                    - cache_dir (str, default: None)
                        Path where caches will be saved locally. If None, will default to
                        ~/.tfutils/<host:post>/<dbname>/<collname>/<exp_id>.
            - load_params (dict)
                Similar to save_params, if you want loading to happen from a different
                location than where saving occurs.   Parameters include:
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
                    - do_restore (bool, default: True)
                        Whether to restore from saved model
                    - load_query (dict)
                        mongodb query describing how to load from loading database
            - sess (tesorflow.Session)
                Object in which to run calculations.  This is required if actual loading/
                saving is going to be done (as opposed to just e.g. getting elements from
                the MongoDB).
            - global_step (tensorflow.Variable)
                Global step variable, the one that is updated by apply_gradients.  This
                is required if being using in a training context.
            - *tfsaver_args, **tsaver_kwargs
                Additional arguments to be passed onto base Saver class constructor
        """

        self.params = params
        self.sonified_params = sonify(self.params)
        self.save_params = save_params
        self.load_params = load_params
        self.sess = sess
        self.global_step = global_step
        self.tfsaver_args = tfsaver_args
        self.tfsaver_kwargs = tfsaver_kwargs

        if save_params is None:
            save_params = {}
        if load_params is None:
            load_params = {}
        location_variables = ['host', 'port', 'dbname', 'collname', 'exp_id']
        for _k in location_variables:
            if _k in save_params:
                sv = save_params[_k]
            else:
                sv = load_params[_k]
            if _k in load_params:
                lv = load_params[_k]
            else:
                lv = save_params[_k]
            setattr(self, _k, sv)
            setattr(self, 'load_' + _k, lv)
        self.sameloc = all([getattr(self, _k) == getattr(self, 'load_' + _k) for _k in location_variables])
        if 'query' in load_params and not load_params['query'] is None and 'exp_id' in load_params['query']:
            self.sameloc = self.sameloc & (load_params['query']['exp_id'] == self.exp_id)

        for _k in ['do_save', 'save_metrics_freq', 'save_valid_freq', 'cache_filters_freq',
                   'save_filters_freq', 'save_initial_filters', 'save_to_gfs']:
            setattr(self, _k, save_params.get(_k, DEFAULT_SAVE_PARAMS[_k]))

        for _k in ['do_restore']:
            setattr(self, _k, load_params.get(_k, DEFAULT_LOAD_PARAMS[_k]))

        self.rec_to_save = None
        self.checkpoint_thread = None
        self.outrecs = []

        self.conn = pymongo.MongoClient(host=self.host, port=self.port)
        self.conn.server_info()
        self.collfs = gridfs.GridFS(self.conn[self.dbname], self.collname)

        if 'port_rec' in save_params:
            tmp_host = save_params.get('host_rec', self.host)
            tmp_port = save_params['port_rec']

            self.conn_rec = pymongo.MongoClient(host=tmp_host, port=tmp_port)
            self.conn_rec.server_info()
            self.collfs_rec = gridfs.GridFS(self.conn_rec[self.dbname], self.collname)
        else:
            self.conn_rec = None
            self.collfs_rec = None

        recent_name = '_'.join([self.dbname, self.collname, self.exp_id, '__RECENT'])
        self.collfs_recent = gridfs.GridFS(self.conn[recent_name])

        self.load_data = None
        load_query = load_params.get('query')
        if load_query is None:
            load_query = {}
        else:
            if self.sameloc and (not save_params == {}):
                raise Exception('Loading pointlessly')
            else:
                self.sameloc = False
                #print('Set sameloc to False!')

        if 'exp_id' not in load_query:
            load_query.update({'exp_id': self.load_exp_id})

        self.load_query = load_query
        if self.load_host != self.host or self.port != self.load_port:
            self.load_conn = pymongo.MongoClient(host=self.load_host,
                                                 port=self.load_port)
            self.load_conn.server_info()
        else:
            self.load_conn = self.conn
        self.load_collfs = gridfs.GridFS(self.load_conn[self.load_dbname],
                                         self.load_collname)
        load_recent_name = '_'.join([self.load_dbname,
                                     self.load_collname,
                                     self.load_exp_id,
                                     '__RECENT'])
        self.load_collfs_recent = gridfs.GridFS(self.load_conn[load_recent_name])

        if 'cache_dir' in save_params:
            cache_dir = save_params['cache_dir']
        else:
            cache_dir = None

        if cache_dir is None:
            self.cache_dir = os.path.join(TFUTILS_HOME,
                                          '%s:%d' % (self.host, self.port),
                                          self.dbname,
                                          self.collname,
                                          self.exp_id)
        else:
            self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

    def load_rec(self):
        # first try and see if anything with the save data exists, since obviously
        # we dont' want to keep loading from the original load location if some work has
        # already been done
        load = self.load_from_db({'exp_id': self.exp_id},
                                 cache_filters=True)
        # if not, try loading from the loading location
        if not load and not self.sameloc:
            load = self.load_from_db(self.load_query,
                                     cache_filters=True,
                                     collfs=self.load_collfs,
                                     collfs_recent=self.load_collfs_recent)
            if load is None:
                raise Exception('You specified load parameters but no record was found with the given spec.')
        self.load_data = load

    def initialize(self, no_scratch=False):
        """
        Fetches record then uses tf's saver.restore
        """
        # fetch record from database and get the filename info from record
        tf_saver = self.tf_saver
        if self.do_restore:
            if self.load_data is None:
                self.load_rec()
            if self.load_data is not None:
                rec, cache_filename = self.load_data
                # get variables to restore
                restore_vars = self.get_restore_vars(cache_filename)
                log.info('Restored Vars:\n' + str([restore_var.name for restore_var in restore_vars]))
                tf_saver_restore = tf.train.Saver(restore_vars)
                # tensorflow restore
                log.info('Restoring variables from record %s (step %d)...' % (str(rec['_id']), rec['step']))
                tf_saver_restore.restore(self.sess, cache_filename)
                log.info('... done restoring.')
                all_variables = tf.global_variables() + tf.local_variables()  # get list of all variables
                unrestored_vars = [var for var in all_variables
                                   if var not in restore_vars]            # compute list of variables not restored
                self.sess.run(tf.variables_initializer(unrestored_vars))  # initialize variables not restored
                assert len(self.sess.run(tf.report_uninitialized_variables())) == 0, self.sess.run(tf.report_uninitialized_variables())
        if (not self.do_restore or self.load_data is None) and not no_scratch:
            init_op_global = tf.global_variables_initializer()
            self.sess.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            self.sess.run(init_op_local)
            log.info('Model variables initialized from scratch.')

    def get_restore_vars(self, save_file):
        """
        Creates list of variables to restore from save_file

        Extracts the subset of variables from tf.global_variables that match the
        name and shape of variables saved in the checkpoint file, and returns these
        as a list of variables to restore.

        Args:
            save_file: path of tf.train.Saver checkpoint
        """
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        log.info('Saved Vars:\n'+str(saved_shapes.keys()))
        var_names = sorted([(var.name.split(':')[0],var) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        for saved_var_name, var in var_names:
            curr_var = var
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
        return restore_vars

    @property
    def tf_saver(self):
        if not hasattr(self, '_tf_saver'):
            self._tf_saver = tf.train.Saver(*self.tfsaver_args, **self.tfsaver_kwargs)
        return self._tf_saver

    def load_from_db(self,
                     query,
                     cache_filters=False,
                     collfs=None,
                     collfs_recent=None):
        """
        Loads checkpoint from the database

        Checks the recent and regular checkpoint fs to find the latest one
        matching the query. Returns the GridOut obj corresponding to the
        record.

        Args:
            query: dict expressing MongoDB query
        """
        if collfs is None:
            collfs = self.collfs
        coll = collfs._GridFS__files
        if collfs_recent is None:
            collfs_recent = self.collfs_recent
        coll_recent = collfs_recent._GridFS__files

        query['saved_filters'] = True
        count = collfs.find(query).count()
        if count > 0:  # get latest that matches query
            ckpt_record = coll.find(query, sort=[('uploadDate', -1)])[0]
            loading_from = coll
        else:
            ckpt_record = None

        try: 
            count_recent = collfs_recent.find(query).count()
        except Exception as inst:
            raise er.OperationFailure(inst.args[0] + "\n Is your dbname too long? Mongo requires that dbnames be no longer than 64 characters.")
        if count_recent > 0:  # get latest that matches query
            ckpt_record_recent = coll_recent.find(query,
                                                  sort=[('uploadDate', -1)])[0]
            # use the record with latest timestamp
            if ckpt_record is None or ckpt_record_recent['uploadDate'] > ckpt_record['uploadDate']:
                loading_from = coll_recent
                ckpt_record = ckpt_record_recent

        if count + count_recent == 0:  # no matches for query
            log.warning('No matching checkpoint for query "{}"'.format(repr(query)))
            return

        database = loading_from._Collection__database
        log.info('Loading checkpoint from %s' % loading_from.full_name)

        if cache_filters:
            filename = os.path.basename(ckpt_record['filename'])
            cache_filename = os.path.join(self.cache_dir, filename)

            # check if there is no local copy
            if not os.path.isfile(cache_filename):
                log.info('No cache file at %s, loading from DB' % cache_filename)
                # create new file to write from gridfs
                load_dest = open(cache_filename, "w+")
                load_dest.close()
                load_dest = open(cache_filename, 'rwb+')
                fsbucket = gridfs.GridFSBucket(database,
                                               bucket_name=loading_from.name.split('.')[0])
                fsbucket.download_to_stream(ckpt_record['_id'], load_dest)
                if ckpt_record['_saver_write_version'] == saver_pb2.SaverDef.V2:
                    assert cache_filename.endswith('.tar')
                    tar = tarfile.open(cache_filename)
                    tar.extractall(path=self.cache_dir)
                    tar.close()
                    cache_filename = os.path.splitext(cache_filename)[0]
                    verify_pb2_v2_files(cache_filename, ckpt_record)
            else:
                if ckpt_record['_saver_write_version'] == saver_pb2.SaverDef.V2:
                    cache_filename = os.path.splitext(cache_filename)[0]
                    verify_pb2_v2_files(cache_filename, ckpt_record)
                log.info('Cache file found at %s, using that to load' % cache_filename)
        else:
            cache_filename = None
        return ckpt_record, cache_filename

    def save(self, train_res=None, valid_res=None, step=None, validation_only=False):
        """
        Actually saves record into DB and makes local filter caches

        """
        if train_res is None:
            train_res = {}
        if valid_res is None:
            valid_res = {}

        if (not validation_only) and (step is None):
            if not hasattr(self.global_step, 'eval'):
                raise NoGlobalStepError('If step is none, you must pass global_step'
                                        ' tensorflow operation to the saver.')
            step = self.global_step.eval(session=self.sess)

        train_res = copy.copy(train_res)
        valid_res = {_k: copy.copy(_v) for _k, _v in valid_res.items()}
        duration = time.time() - self.start_time_step

        if self.rec_to_save is None:
            rec = {'exp_id': self.exp_id,
                   'params': self.sonified_params,
                   'saved_filters': False,
                   'duration': duration}
            self.rec_to_save = rec
        else:
            rec = self.rec_to_save
        rec['step'] = step

        if len(train_res) > 0:
            # TODO: also include error rate of the train set to monitor overfitting
            message = 'Step {} ({:.0f} ms) -- '.format(step, 1000 * duration)
            msg2 = ['{}: {:.4f}'.format(k, v) for k, v in train_res.items() if k != 'optimizer' and k not in self.save_to_gfs]
            message += ', '.join(msg2)
            log.info(message)

            if 'optimizer' in train_res:
                del train_res['optimizer']
            if 'train_results' not in rec:
                rec['train_results'] = []
            rec['train_results'].append(train_res)

        # print validation set performance
        if len(valid_res) > 0:
            rec['validation_results'] = valid_res
            message = 'Validation -- '
            message += ', '.join('{}: {}'.format(k,
                                                 {_k: _v for _k, _v in v.items() if _k not in self.save_to_gfs}
                                                 ) for k, v in valid_res.items())
            log.info(message)

        if validation_only:
            rec['validates'] = self.load_data[0]['_id']
            save_filters_permanent = save_filters_tmp = False
            need_to_save = True
        else:
            save_filters_permanent = ((step % self.save_filters_freq == 0) and
                                      (step > 0 or (self.save_initial_filters and not self.load_data)))
            save_filters_tmp = ((step % self.cache_filters_freq == 0) and
                                (step > 0 or (self.save_initial_filters and not self.load_data)))
            save_metrics_now = step % self.save_metrics_freq == 0
            save_valid_now = step % self.save_valid_freq == 0
            need_to_save = save_filters_permanent or save_filters_tmp or save_metrics_now or save_valid_now

        need_to_save = self.do_save and need_to_save

        if need_to_save:
            self.rec_to_save = None
            self.sync_with_host()
            save_to_gfs = {}
            for _k in self.save_to_gfs:
                if train_res:
                    if 'train_results' not in save_to_gfs:
                        save_to_gfs['train_results'] = {}
                    if _k in train_res:
                        save_to_gfs['train_results'][_k] = [r.pop(_k) for r in rec['train_results'] if _k in r]
                        if len(save_to_gfs['train_results'][_k]) == 1:
                            save_to_gfs['train_results'][_k] == save_to_gfs['train_results'][_k][0]
                if valid_res:
                    if 'validation_results' not in save_to_gfs:
                        save_to_gfs['validation_results'] = {}
                    for _vk in valid_res:
                        if _vk not in save_to_gfs['validation_results']:
                            save_to_gfs['validation_results'][_vk] = {}
                        if _k in valid_res[_vk]:
                            save_to_gfs['validation_results'][_vk][_k] = valid_res[_vk].pop(_k)

            save_rec = sonify(rec)
            make_mongo_safe(save_rec)

            thread = threading.Thread(target=self._save_thread,
                                      args=(save_filters_permanent,
                                            save_filters_tmp,
                                            save_rec,
                                            step,
                                            save_to_gfs))
            thread.daemon = True
            thread.start()
            self.checkpoint_thread = thread

    def sync_with_host(self):
        if self.checkpoint_thread is not None:
            self.checkpoint_thread.join()
            self.checkpoint_thread = None

    def _save_thread(self, save_filters_permanent, save_filters_tmp, save_rec, step, save_to_gfs):

        if self.collfs_rec:
            save_rec['saved_filters'] = False
            log.info('Inserting record into record database.')
            outrec = self.collfs_rec._GridFS__files.insert_one(save_rec)

            if not isinstance(outrec, ObjectId):
                outrec = outrec.inserted_id

            if save_to_gfs:
                idval = str(outrec)
                save_to_gfs_path = idval + "_fileitems"
                self.collfs_rec.put(cPickle.dumps(save_to_gfs), filename=save_to_gfs_path, item_for=outrec)

        if save_filters_permanent or save_filters_tmp:
            save_rec['saved_filters'] = True
            save_path = os.path.join(self.cache_dir, 'checkpoint')
            log.info('Saving model with path prefix %s ... ' % save_path)
            saved_path = self.tf_saver.save(self.sess,
                                            save_path=save_path,
                                            global_step=step,
                                            write_meta_graph=False)
            log.info('... done saving with path prefix %s' % saved_path)
            putfs = self.collfs if save_filters_permanent else self.collfs_recent
            log.info('Putting filters into %s database' % repr(putfs))
            save_rec['_saver_write_version'] = self.tf_saver._write_version
            if self.tf_saver._write_version == saver_pb2.SaverDef.V2:
                file_data = get_saver_pb2_v2_files(saved_path)
                save_rec['_saver_num_data_files'] = file_data['num_data_files']
                tarfilepath = saved_path + '.tar'
                tar = tarfile.open(tarfilepath, 'w')
                for _f in file_data['files']:
                    tar.add(_f, arcname=os.path.split(_f)[1])
                tar.close()
                with open(tarfilepath, 'rb') as _fp:
                    outrec = putfs.put(_fp, filename=tarfilepath, **save_rec)
            else:
                with open(saved_path, 'rb') as _fp:
                    outrec = putfs.put(_fp, filename=saved_path, **save_rec)
            log.info('... done putting filters into database.')

        if not save_filters_permanent:
            save_rec['saved_filters'] = False
            log.info('Inserting record into database.')
            outrec = self.collfs._GridFS__files.insert_one(save_rec)

        if not isinstance(outrec, ObjectId):
            outrec = outrec.inserted_id

        if save_to_gfs:
            idval = str(outrec)
            save_to_gfs_path = idval + "_fileitems"
            self.collfs.put(cPickle.dumps(save_to_gfs), filename=save_to_gfs_path, item_for=outrec)

        sys.stdout.flush()  # flush the stdout buffer
        self.outrecs.append(outrec)


def predict(step, results):
    if not hasattr(results['output'], '__iter__'):
        outputs = [results['outputs']]
    else:
        outputs = results['outputs']

    preds = [tf.argmax(output, 1) for output in outputs]

    return preds


def run_targets(sess,
                dbinterface,
                target_name,
                target,
                valid_loop,
                num_steps,
                online_agg_func,
                agg_func,
                save_intermediate_freq=None,
                validation_only=False):
    """TODO:  this code resembles train() function, possible want to unify
    """
    agg_res = None

    if save_intermediate_freq is not None:
        n0 = len(dbinterface.outrecs)

    for _step in tqdm.trange(num_steps, desc=target_name):
        if valid_loop is not None:
            res = valid_loop(sess, target)
        else:
            res = sess.run(target)
        assert hasattr(res, 'keys'), 'result must be a dictionary'
        if save_intermediate_freq is not None and _step % save_intermediate_freq == 0:
            dbinterface.save(valid_res={target_name: res},
                             step=_step,
                             validation_only=validation_only)
        agg_res = online_agg_func(agg_res, res, _step)

    result = agg_func(agg_res)

    if save_intermediate_freq is not None:
        dbinterface.sync_with_host()
        n1 = len(dbinterface.outrecs)
        result['intermediate_steps'] = dbinterface.outrecs[n0: n1]

    return result


def run_targets_dict(sess,
                     targets,
                     save_intermediate_freq=None,
                     dbinterface=None,
                     validation_only=False):
    """
    Helper function for actually computing validation results.
    """
    results = {}
    for target_name in targets:
        num_steps = targets[target_name]['num_steps']
        target = targets[target_name]['targets']
        agg_func = targets[target_name]['agg_func']
        online_agg_func = targets[target_name]['online_agg_func']
        valid_loop = targets[target_name]['valid_loop']
        results[target_name] = run_targets(sess,
                                           dbinterface,
                                           target_name,
                                           target,
                                           valid_loop,
                                           num_steps,
                                           online_agg_func,
                                           agg_func,
                                           save_intermediate_freq,
                                           validation_only)
    if dbinterface is not None:
        dbinterface.save(valid_res=results, validation_only=validation_only)
    return results


def start_queues(sess):
    """Helper function for starting queues before running processes.
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    return coord, threads


def stop_queues(sess, queues, coord, threads):
    """Helper function for stopping queues cleanly.
    """
    coord.request_stop()
    coord.join(threads)
    for queue in queues:
        close_op = queue.close(cancel_pending_enqueues=True)
        sess.run(close_op)


def test(sess,
         queues,
         dbinterface,
         valid_targets,
         save_intermediate_freq=None):

    """
    Actually runs the testing evaluation loop.

    :Args:
        - sess: (tensorflow.Session)
            Object in which to run calculations
        - queues (list of CustomQueue)
            Objects containing asynchronously queued data iterators
        - dbinterface (DBInterface object)
            Saver through which to save results
        - valid_targets (dict of tensorflow objects)
            Objects on which validation will be computed
        - save_intermediate_freq (None or int)
            How frequently to save intermediate results captured during test
            None means no intermediate saving will be saved
    """
    coord, threads = start_queues(sess)
    dbinterface.start_time_step = time.time()
    valid_results_summary = run_targets_dict(sess,
                                             valid_targets,
                                             save_intermediate_freq=save_intermediate_freq,
                                             dbinterface=dbinterface,
                                             validation_only=True)
    dbinterface.sync_with_host()
    stop_queues(sess, queues, coord, threads)
    return valid_results_summary, dbinterface.outrecs


def test_from_params_old(load_params,
                     model_params,
                     validation_params,
                     log_device_placement=False,
                     save_params=None,
                     dont_run=False,
                     inter_op_parallelism_threads=40,
                     ):

    """
    Main testing interface function.

    Same as train_from_parameters; but just performs testing without training.

    For documentation, see argument descriptions in train_from_params.
    """
    if save_params is None:
        save_params = {}
    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]

        # create session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=log_device_placement,
            inter_op_parallelism_threads=inter_op_parallelism_threads))

        dbinterface = DBInterface(load_params=load_params)
        dbinterface.load_rec()
        ld = dbinterface.load_data
        assert ld is not None, "No load data found for query, aborting"
        ld = ld[0]
        # TODO: have option to reconstitute model_params entirely from saved object ("revivification")
        model_params['seed'] = ld['params']['model_params']['seed']
        cfg_final = ld['params']['model_params']['cfg_final']
        train_queue_params = ld['params']['train_params']['queue_params']
        params = {'train_params': {'queue_params': train_queue_params}}
        valid_targets_dict, queues = get_valid_targets_dict(validation_params,
                                                            model_params,
                                                            None,
                                                            cfg_final=cfg_final,
                                                            **params)
        model_params['cfg_final'] = cfg_final
        load_params['do_restore'] = True
        params = {'load_params': load_params,
                  'save_params': save_params,
                  'model_params': model_params,
                  'validation_params': validation_params,
                  'log_device_placement': log_device_placement,
                  'inter_op_parallelism_threads': inter_op_parallelism_threads}

        dbinterface = DBInterface(sess=sess,
                                  params=params,
                                  load_params=load_params,
                                  save_params=save_params)
        dbinterface.initialize()

        if dont_run:
            return sess, queues, dbinterface, valid_targets_dict

        save_intermediate_freq = save_params.get('save_intermediate_freq')
        res = test(sess,
                   queues,
                   dbinterface,
                   valid_targets_dict,
                   save_intermediate_freq=save_intermediate_freq)
        sess.close()
        return res


def train(sess,
          mode,
          queues,
          dbinterface,
          train_loop,
          train_targets,
          global_step,
          num_steps=float('inf'),
          thres_loss=DEFAULT_TRAIN_THRES_LOSS,
          validate_first=True,
          validation_targets=None):
    """
    Actually runs the training evaluation loop.

    :Args:
        - sess: (tesorflow.Session)
            Object in which to run calculations
        - queues (list of Queue)
            Objects containing asynchronously queued data iterators
        - dbinterface (DBInterface object)
            Saver through which to save results
        - train_loop (callable withs args: sess and train_targets)
            Callable that specifies a custom training loop
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

    # Collect args in a dict of lists
    train_args = {
        'sess': sess,
        'queues': queues,
        'num_steps': num_steps,
        'thres_loss': thres_loss,
        'train_loop': train_loop,
        'global_step': global_step,
        'dbinterface': dbinterface,
        'train_targets': train_targets,
        'validate_first': validate_first,
        'validation_targets': validation_targets}

    # Convert to a list of dicts
    _trains = [{key: value[i] for (key, value) in train_args.items()}
               for i in range(len(train_targets))]

    num_steps = [t['num_steps'] for t in _trains]
    steps = [t['global_step'].eval(session=t['sess']) for t in _trains]

    # Start queues and initial validation
    for (train, step) in zip(_trains, steps):

        if step >= train['num_steps']:
            log.info('Training cancelled since step ({}) is >= num_steps ({})'.format(step, train['num_steps']))
            return

        log.info('Training beginning ...')
        train['coord'], train['threads'] = start_queues(train['sess'])

        if step == 0:
            train['dbinterface'].start_time_step = time.time()
            if train['validate_first']:
                validation_res = run_targets_dict(train['sess'],
                                                  train['validation_targets'],
                                                  dbinterface=train['dbinterface'])
    # Run training
    while steps < num_steps:

        for (train, step) in zip(_trains, steps):

            old_step = step
            train['dbinterface'].start_time_step = time.time()

            if train['train_loop'] is not None:
                train_results = train['train_loop'](train['sess'],
                                                    train['train_targets'])
            else:
                train_results = train['sess'].run(train['train_targets'])

            step = train['global_step'].eval(session=train['sess'])
            if step <= old_step:
                raise NoChangeError('Your optimizer should have incremented the global step,'
                                    ' but did not: old_step=%d, new_step=%d' % (old_step, step))
            if train_results['loss'] > train['thres_loss']:
                raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_results['loss'],
                                                                                     train['thres_loss']))

            vtargs = train['validation_targets'] if step % train['dbinterface'].save_valid_freq == 0 else {}
            validation_res = run_targets_dict(train['sess'], vtargs)

            train['dbinterface'].save(train_res=train_results,
                                      valid_res=validation_res,
                                      validation_only=False)

        steps = [t['global_step'].eval(session=t['sess']) for t in _trains]

    # Save and close the session
    res = []
    for train in _trains:
        stop_queues(train['sess'], train['queues'], train['coord'], train['threads'])
        train['dbinterface'].sync_with_host()
        res.append(train['dbinterface'].outrecs)
    _trains[0]['sess'].close()
    return res


def train_from_params_old(save_params,
                          model_params,
                          train_params,
                          loss_params=None,
                          learning_rate_params=None,
                          optimizer_params=None,
                          validation_params=None,
                          postsess_params=None,
                          log_device_placement=False,
                          load_params=None,
                          dont_run=False,
                          inter_op_parallelism_threads=40,
                          ):

    model_params = [model_params] if not isinstance(model_params,
                                                    list) else model_params
    num_models = len(model_params)
    list_lens = [num_models]

    params = OrderedDict({
        'model_params': model_params,
        'train_params': train_params,
        'validation_params': validation_params,
        'save_params': save_params,
        'load_params': load_params,
        'loss_params': loss_params,
        'optimizer_params': optimizer_params,
        'learning_rate_params': learning_rate_params,
        'log_device_placement': log_device_placement,
        'inter_op_parallelism_threads': inter_op_parallelism_threads})

    # Ensure params is a dict of lists, using defaults when necessary.
    for name, param_list in params.items():
        if not param_list:
            param_list = DEFAULT_PARAMS[name]
        if not isinstance(param_list, list):
            param_list = [copy.deepcopy(param_list) for _ in range(num_models)]
        if len(param_list) != num_models and len(param_list) == 1:
            param_list += (num_models - 1) * copy.deepcopy(param_list)

        for model_num, param in enumerate(param_list):
            if name == 'model_params':
                if 'train' not in param:
                    param['train'] = True
                if 'prefix' not in param:
                    param['prefix'] = 'model_{}'.format(model_num)
                if 'devices' not in param:
                    param['devices'] = [DEFAULT_DEVICES.pop()]
                if all([isinstance(d, int) for d in param['devices']]):
                    param['devices'] = map('/gpu:{}'.format, param['devices'])
                if 'num_gpus' not in param:
                    param['num_gpus'] = len(param['devices'])
                assert param['num_gpus'] == len(param['devices'])

            if name == 'train_params':
                if 'num_steps' not in param:
                    param['num_steps'] = DEFAULT_TRAIN_NUM_STEPS
                if 'thres_loss' not in param:
                    param['thres_loss'] = DEFAULT_TRAIN_THRES_LOSS
                if 'train_loop' not in param:
                    param['train_loop'] = {'func': None}
                if 'validate_first' not in param:
                    param['validate_first'] = True

        params[name] = param_list

        list_lens.append(len(param_list))
        assert isinstance(param_list, list), '{} should also be a list'.format(name)
        assert len(param_list) == num_models, '{} should have length'.format(num_models)
    assert len(np.unique(list_lens)) == 1, 'All param lists should have be same length!'

    # Prepare args to be passed to `base.train`.
    train_args = {
        'sess': num_models * [None],
        'mode': num_models * ['train'],
        'queues': num_models * [None],
        'dbinterface': num_models * [None],
        'global_step': num_models * [None],
        'train_targets': [dict() for _ in range(num_models)],
        'validation_targets': [dict() for _ in range(num_models)],
        'num_steps': [p['num_steps'] for p in params['train_params']],
        'thres_loss': [p['thres_loss'] for p in params['train_params']],
        'train_loop': [p['train_loop']['func'] for p in params['train_params']],
        'validate_first': [p['validate_first'] for p in params['train_params']]}

    # Generate and distribute graph, returning targets
    with tf.Graph().as_default():
        params, train_args = get_targets(params, train_args)

    if dont_run:
        return train_args

    return train(**train_args)


def get_targets(params, train_args):

    # For convenience, use list of dicts instead of dict of lists
    _params = [{key: value[i] for (key, value) in params.items()}
               for i in range(len(params['model_params']))]
    _trains = [{key: value[i] for (key, value) in train_args.items()}
               for i in range(len(params['model_params']))]

    # Build a graph for each distinct model.
    for param, train in zip(_params, _trains):
        with tf.variable_scope(param['model_params']['prefix']):

            train['global_step'] = tf.get_variable('global_step', [],
                                                   dtype=tf.int64, trainable=False,
                                                   initializer=tf.constant_initializer(0))
            (param['learning_rate_params'],
             learning_rate) = get_learning_rate(train['global_step'],
                                                **param['learning_rate_params'])
            (param['optimizer_params'],
             optimizer_base) = get_optimizer_base(learning_rate,
                                                  param['optimizer_params'])
            (param['train_params']['data_params'],
             train['queues'],
             inputs) = get_data(queue_params=param['train_params']['queue_params'],
                                **param['train_params']['data_params'])
            tower_losses = []
            tower_grads = []
            devices = param['model_params']['devices']
            inputs = split_input(inputs, param['model_params']['num_gpus'])

            # Distribute graph across desired devices.
            for device, input in zip(devices, inputs):
                with tf.device(device), tf.name_scope('gpu_' + device[-1]):

                    (param['model_params'],
                     output) = get_model(input, **param['model_params'])

                    if param['train_params'].get('targets') is not None:
                        ttargs = copy.deepcopy(param['train_params']['targets'])
                        ttargs_func = ttargs.pop('func')
                        ttarg = ttargs_func(input, output, **ttargs)
                        train['train_targets'].update(ttarg)

                    (param['loss_params'],
                     loss) = get_loss(input, output, **param['loss_params'])

                    tf.get_variable_scope().reuse_variables()

                    grad = optimizer_base.compute_gradients(loss)
                    tower_losses.append(loss)
                    tower_grads.append(grad)

                    (train['validation_targets'],
                     vqueue) = get_valid_targets_dict(default_queue_params=param['train_params']['queue_params'],
                                                      **param)
                    train['queues'].extend(vqueue)

        # Accumulate and average gradients on the host.
        with tf.device('/cpu:0'), tf.variable_scope(param['model_params']['prefix']):
            loss = tf.reduce_mean(tf.stack(tower_losses))
            average_grads = average_gradients(tower_grads)
            optimizer = optimizer_base.apply_gradients(average_grads,
                                                       train['global_step'])

            train['train_targets'].update({'loss': loss,
                                           'optimizer': optimizer,
                                           'learning_rate': learning_rate})
    # Create session.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                      log_device_placement=param['log_device_placement'],
                      inter_op_parallelism_threads=param['inter_op_parallelism_threads']))

    init_op_global = tf.global_variables_initializer()
    sess.run(init_op_global)
    init_op_local = tf.local_variables_initializer()
    sess.run(init_op_local)
    log.info('Initialized from scratch first')

    for param, train in zip(_params, _trains):

        var_list = {}
        all_vars = variables._all_saveable_objects()
        prefix_str = param['model_params']['prefix'] + '/'

        for each_var in all_vars:
            if each_var.op.name.startswith(prefix_str):
                new_name = each_var.op.name[len(prefix_str):]
                if new_name.startswith(prefix_str):
                    new_name = new_name[len(prefix_str):]
                var_list[new_name] = each_var

        train['dbinterface'] = DBInterface(sess=sess,
                                           params=param,
                                           # var_list=var_list,
                                           global_step=train['global_step'],
                                           save_params=param['save_params'],
                                           load_params=param['load_params'])
        train['dbinterface'].initialize(no_scratch=True)
        # train['dbinterface'].initialize()
        train['sess'] = sess

    # Convert back to a dictionary of lists
    params = {key: [param[key] for param in _params]
              for key in _params[0].keys()}
    train_args = {key: [train[key] for train in _trains]
                  for key in _trains[0].keys()}

    return params, train_args


def get_valid_targets_dict(validation_params,
                           model_params,
                           loss_params,
                           default_queue_params,
                           cfg_final=None,
                           **params):
    """Helper function for creating validation target operations.
       NB: this function may modify validation_params"""
    valid_targets_dict = OrderedDict()
    queues = []
    model_params = copy.deepcopy(model_params)
    model_params.pop('train', None)  # hackety-hack
    if cfg_final is None:
        assert 'cfg_final' in model_params
        cfg_final = model_params['cfg_final']
    assert 'seed' in model_params
    for vtarg in validation_params:
        queue_params = validation_params[vtarg].get('queue_params', default_queue_params)

        _, queue, vinputs = get_data(queue_params=queue_params,
                                     **validation_params[vtarg]['data_params'])
        queues.extend(queue)
        scope_name = 'validation/%s' % vtarg
        with tf.name_scope(scope_name):
            _mp, voutputs = get_model(vinputs, train=False, **model_params)
            check_model_equivalence(_mp['cfg_final'], cfg_final, scope_name)
            tf.get_variable_scope().reuse_variables()
        validation_params[vtarg], valid_targets_dict[vtarg] = get_validation_target(vinputs, voutputs,
                                                                                    **validation_params[vtarg])

    return valid_targets_dict, queues


def check_model_equivalence(m1, m2, name):
    """TODO: fill this in to make it stronger"""
    assert set(m1.keys()) == set(m2.keys()), (m1.keys(), m2.keys())


def get_validation_target(vinputs, voutputs,
                          default_target_func=utils.get_loss_dict,
                          default_target_params=DEFAULT_LOSS_PARAMS,
                          default_loop_func=None,
                          default_loop_params=DEFAULT_LOOP_PARAMS,
                          agg_func=utils.identity_func,
                          online_agg_func=utils.append_and_return,
                          **validation_params):
    target_params = validation_params.get('targets', dict(default_target_params))
    target_func = target_params.pop('func', default_target_func)
    vtargets = target_func(vinputs, voutputs, **target_params)
    target_params['func'] = target_func
    validation_params['targets'] = target_params

    valid_loop_params = validation_params.get('valid_loop', dict(default_loop_params))
    valid_loop_func = valid_loop_params.pop('func', default_loop_func)
    valid_loop = valid_loop_func
    valid_loop_params['func'] = valid_loop_func
    validation_params['valid_loop'] = valid_loop_params

    if 'num_steps' not in validation_params:
        assert hasattr(vinputs, 'total_batches'), '"num_batches" not specified in validation params, '\
            'data object must have "total_batches" attribute to be used as default.'
        validation_params['num_steps'] = vinputs.total_batches
    validation_params['agg_func'] = agg_func
    validation_params['online_agg_func'] = online_agg_func
    valid_targets = {'targets': vtargets,
                     'valid_loop': valid_loop,
                     'agg_func': validation_params['agg_func'],
                     'online_agg_func': validation_params['online_agg_func'],
                     'num_steps': validation_params['num_steps'],
                     }
    return validation_params, valid_targets


def get_data(func, queue_params=None, **data_params):
    data_provider = func(**data_params)
    input_ops = data_provider.init_ops()
    assert len(input_ops) == data_params['n_threads'], (len(input_ops), data_params['n_threads'])
    assert len(input_ops) > 0, len(input_ops)
    batch_size = data_params['batch_size']
    data_params['func'] = func
    enqueue_ops = []
    queue = get_queue(input_ops[0], **queue_params)
    for input_op in input_ops:
        # enqueue_ops.append(queue.enqueue_many(input_op))
        if batch_size == 1:
            enqueue_ops.append(queue.enqueue(input_op))
        else:
            enqueue_ops.append(queue.enqueue_many(input_op))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue,
                                                                             enqueue_ops))
    if queue_params['batch_size'] == 1:
        inputs = queue.dequeue()
    else:
        inputs = queue.dequeue_many(queue_params['batch_size'])
    return data_params, [queue], inputs


def split_input(inputs, num_gpus=1):
    if num_gpus == 1:
        return [inputs]

    temp_args = {v: tf.split(inputs[v], axis=0, num_or_size_splits=num_gpus) for v in inputs}
    list_of_args = [{now_arg: temp_args[now_arg][ind] for now_arg in temp_args} for ind in xrange(num_gpus)]

    return list_of_args


def get_model(inputs, func, seed=0, train=False, **model_params):
    model_params['seed'] = seed
    model_params['train'] = train
    outputs, cfg_final = func(inputs=inputs,
                              **model_params)
    model_params['func'] = func
    model_params['cfg_final'] = cfg_final
    return model_params, outputs


def get_loss(train_inputs,
             train_outputs,
             targets=DEFAULT_LOSS_PARAMS['targets'],
             agg_func=DEFAULT_LOSS_PARAMS['agg_func'],
             loss_per_case_func=DEFAULT_LOSS_PARAMS['loss_per_case_func'],
             **loss_params):
    loss_params['targets'] = targets
    loss_params['agg_func'] = agg_func
    loss_params['loss_per_case_func'] = loss_per_case_func
    loss = utils.get_loss(train_inputs, train_outputs, **loss_params)
    return loss_params, loss


def get_learning_rate(global_step,
                      func=tf.train.exponential_decay,
                      **learning_rate_params):
    learning_rate = func(global_step=global_step,
                         **learning_rate_params)
    learning_rate_params['func'] = func
    return learning_rate_params, learning_rate


def get_optimizer(learning_rate,
                  loss,
                  global_step,
                  optimizer_params,
                  default_optimizer_params=DEFAULT_OPTIMIZER_PARAMS,
                  default_optimizer_func=ClipOptimizer):
    if optimizer_params is None:
        optimizer_params = dict(default_optimizer_params)
    func = optimizer_params.pop('func', default_optimizer_func)
    optimizer_base = func(learning_rate=learning_rate,
                          **optimizer_params)
    optimizer = optimizer_base.minimize(loss, global_step)
    optimizer_params['func'] = func
    return optimizer_params, optimizer


def get_optimizer_base(learning_rate,
                       optimizer_params,
                       default_optimizer_params=DEFAULT_OPTIMIZER_PARAMS,
                       default_optimizer_func=ClipOptimizer):
    if optimizer_params is None:
        optimizer_params = dict(default_optimizer_params)
    func = optimizer_params.pop('func', default_optimizer_func)
    optimizer_base = func(learning_rate=learning_rate,
                          **optimizer_params)
    optimizer_params['func'] = func
    return optimizer_params, optimizer_base


def average_gradients(tower_grads):
    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        # print(grads_and_vars)
        grads = []
        for g, _ in grads_and_vars:
            # print(g.get_shape().as_list(), g)
            grads.append(tf.expand_dims(g, axis=0))
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)
        # all variables are the same so we just use the first gpu variables
        var = grads_and_vars[0][1]
        grad_and_var = (grad, var)
        average_grads.append(grad_and_var)
    return average_grads


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


"""
Something like this could be used to create and save variables
in a readable format.
    def save_variables_to_readable_format():
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
