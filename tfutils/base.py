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
from tfutils.utils import (sonify,
                           make_mongo_safe,
                           aggregate_outputs,
                           get_saver_pb2_v2_files,
                           verify_pb2_v2_files,
                           frozendict,
                           format_devices,
                           strip_prefix)

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

DEFAULT_MODEL_SEED = 0
DEFAULT_MODEL_TRAIN = False
DEFAULT_MODEL_PARAMS = frozendict({'seed': DEFAULT_MODEL_SEED,
                                   'train': DEFAULT_MODEL_TRAIN})
DEFAULT_DONT_RUN = False
DEFAULT_SKIP_CHECK = False
DEFAULT_LOG_DEVICE_PLACEMENT = False
DEFAULT_INTER_OP_PARALLELISM_THREADS = 40
DEFAULT_TRAIN_NUM_STEPS = None
DEFAULT_TRAIN_THRES_LOSS = 100

DEFAULT_HOST = '/cpu:0'
DEFAULT_DEVICES = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
DEFAULT_LOOP_PARAMS = frozendict()
DEFAULT_LOAD_PARAMS = frozendict({'do_restore': True})
DEFAULT_LEARNING_RATE_PARAMS = frozendict({'func': tf.train.exponential_decay})

DEFAULT_LOSS_PARAMS = frozendict({'targets': ['labels'],
                                  'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
                                  'agg_func': tf.reduce_mean})

DEFAULT_OPTIMIZER_PARAMS = frozendict({'optimizer_class': tf.train.MomentumOptimizer,
                                       'momentum': 0.9})

DEFAULT_SAVE_PARAMS = frozendict({'save_metrics_freq': 100,
                                  'save_valid_freq': 3000,
                                  'cache_filters_freq': 3000,
                                  'save_filters_freq': 30000,
                                  'save_initial_filters': True,
                                  'save_to_gfs': (),
                                  'do_save': True})

DEFAULT_PARAMS = frozendict({
    'dont_run': False,
    'skip_check': False,
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
        self._skip_check = params.get('skip_check', False)
        if self._skip_check:
            log.warning('Skipping version check and info...')
        self.sonified_params = sonify(self.params, skip=self._skip_check)
        self.save_params = save_params
        self.load_params = load_params
        self.sess = sess
        self.global_step = global_step
        self.tfsaver_args = tfsaver_args
        self.tfsaver_kwargs = tfsaver_kwargs
        self.var_list = tfsaver_kwargs.get('var_list', None)

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
        self.sameloc = all([getattr(self, _k) == getattr(
            self, 'load_' + _k) for _k in location_variables])
        if 'query' in load_params and not load_params['query'] is None and 'exp_id' in load_params['query']:
            self.sameloc = self.sameloc & (
                load_params['query']['exp_id'] == self.exp_id)

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
            self.collfs_rec = gridfs.GridFS(
                self.conn_rec[self.dbname], self.collname)
        else:
            self.conn_rec = None
            self.collfs_rec = None

        recent_name = '_'.join(
            [self.dbname, self.collname, self.exp_id, '__RECENT'])
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
                # print('Set sameloc to False!')

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
        self.load_collfs_recent = gridfs.GridFS(
            self.load_conn[load_recent_name])

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
                raise Exception(
                    'You specified load parameters but no record was found with the given spec.')
        self.load_data = load

    def initialize(self, no_scratch=False):
        """Fetch record then uses tf's saver.restore."""
        # fetch record from database and get the filename info from record
        tf_saver = self.tf_saver
        if self.do_restore:
            if self.load_data is None:
                self.load_rec()
            if self.load_data is not None:
                rec, cache_filename = self.load_data
                # get variables to restore
                if self.var_list is None:
                    restore_vars = self.get_restore_vars(cache_filename)
                    log.info('Restored Vars:\n' + str([restore_var.name for restore_var in restore_vars]))
                else:
                    restore_vars = self.var_list
                    log.info('Restored Vars:\n' + str([restore_var for restore_var in restore_vars]))
                tf_saver_restore = tf.train.Saver(restore_vars)

                # tensorflow restore
                log.info('Restoring variables from record %s (step %d)...' % (str(rec['_id']), rec['step']))
                tf_saver_restore.restore(self.sess, cache_filename)
                log.info('... done restoring.')
                if self.var_list is None:
                    all_variables = tf.global_variables() + tf.local_variables()  # get list of all variables
                    unrestored_vars = [var for var in all_variables
                                       if var not in restore_vars]  # compute list of variables not restored
                    self.sess.run(tf.variables_initializer(unrestored_vars))  # initialize variables not restored
                    assert len(self.sess.run(tf.report_uninitialized_variables())) == 0, self.sess.run(tf.report_uninitialized_variables())
        if (not self.do_restore or self.load_data is None) and not no_scratch:
            init_op_global = tf.global_variables_initializer()
            self.sess.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            self.sess.run(init_op_local)
            log.info('Model variables initialized from scratch.')

    def get_restore_vars(self, save_file):
        """Create list of variables to restore from save_file.

        Extracts the subset of variables from tf.global_variables that match the
        name and shape of variables saved in the checkpoint file, and returns these
        as a list of variables to restore.

        Args:
            save_file: path of tf.train.Saver checkpoint
        """
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        log.info('Saved Vars:\n' + str(saved_shapes.keys()))
        var_names = sorted([(var.name.split(':')[0], var) for var in tf.global_variables()
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
            self._tf_saver = tf.train.Saver(
                *self.tfsaver_args, **self.tfsaver_kwargs)
        return self._tf_saver

    def load_from_db(self,
                     query,
                     cache_filters=False,
                     collfs=None,
                     collfs_recent=None):
        """Load checkpoint from the database.

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
            raise er.OperationFailure(inst.args[
                                      0] + "\n Is your dbname too long? Mongo requires that dbnames be no longer than 64 characters.")
        if count_recent > 0:  # get latest that matches query
            ckpt_record_recent = coll_recent.find(query,
                                                  sort=[('uploadDate', -1)])[0]
            # use the record with latest timestamp
            if ckpt_record is None or ckpt_record_recent['uploadDate'] > ckpt_record['uploadDate']:
                loading_from = coll_recent
                ckpt_record = ckpt_record_recent

        if count + count_recent == 0:  # no matches for query
            log.warning(
                'No matching checkpoint for query "{}"'.format(repr(query)))
            return

        database = loading_from._Collection__database
        log.info('Loading checkpoint from %s' % loading_from.full_name)

        if cache_filters:
            filename = os.path.basename(ckpt_record['filename'])
            cache_filename = os.path.join(self.cache_dir, filename)

            # check if there is no local copy
            if not os.path.isfile(cache_filename):
                log.info('No cache file at %s, loading from DB' %
                         cache_filename)
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
                log.info('Cache file found at %s, using that to load' %
                         cache_filename)
        else:
            cache_filename = None
        return ckpt_record, cache_filename

    def save(self, train_res=None, valid_res=None, step=None, validation_only=False):
        """Actually save record into DB and makes local filter caches."""
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
            # TODO: also include error rate of the train set to monitor
            # overfitting
            message = 'Step {} ({:.0f} ms) -- '.format(step, 1000 * duration)
            msg2 = ['{}: {:.4f}'.format(k, v) for k, v in train_res.items()
                    if k not in ['optimizer', 'grads', 'zero_grad'] and k not in self.save_to_gfs]
            message += ', '.join(msg2)
            log.info(message)

            if 'grads' in train_res:
                del train_res['grads']
            if 'zero_grad' in train_res:
                del train_res['zero_grad']
            if 'optimizer' in train_res:
                del train_res['optimizer']
            if 'train_results' not in rec:
                rec['train_results'] = []
            rec['train_results'].append(train_res)

        # print validation set performance
        if len(valid_res) > 0:
            rec['validation_results'] = valid_res
            message = 'Validation -- '
            message += ', '.join('{}: {}'.format(k, {_k: _v for _k, _v in v.items()
                                                     if _k not in self.save_to_gfs})
                                                for k, v in valid_res.items())
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
                        save_to_gfs['train_results'][_k] = [
                            r.pop(_k) for r in rec['train_results'] if _k in r]
                        if len(save_to_gfs['train_results'][_k]) == 1:
                            save_to_gfs['train_results'][
                                _k] == save_to_gfs['train_results'][_k][0]
                if valid_res:
                    if 'validation_results' not in save_to_gfs:
                        save_to_gfs['validation_results'] = {}
                    for _vk in valid_res:
                        if _vk not in save_to_gfs['validation_results']:
                            save_to_gfs['validation_results'][_vk] = {}
                        if _k in valid_res[_vk]:
                            save_to_gfs['validation_results'][
                                _vk][_k] = valid_res[_vk].pop(_k)

            save_rec = sonify(rec, skip=self._skip_check)
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

    # TODO: throw error if...
    # TODO: Look into python threading lib.
    # TODO: Talk to Damian if questions.
    def sync_with_host(self):
        if self.checkpoint_thread is not None:
            self.checkpoint_thread.join()  # Why no error if error in thread??
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
                self.collfs_rec.put(cPickle.dumps(
                    save_to_gfs), filename=save_to_gfs_path, item_for=outrec)

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
            self.collfs.put(cPickle.dumps(save_to_gfs),
                            filename=save_to_gfs_path, item_for=outrec)

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
    """TODO:  this code resembles train() function, possible want to unify."""
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
    """Helper function for actually computing validation results."""
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
    """Helper function for starting queues before running processes."""
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    return coord, threads


def stop_queues(sess, queues, coord, threads):
    """Helper function for stopping queues cleanly."""
    coord.request_stop()
    coord.join(threads)
    for queue in queues:
        close_op = queue.close(cancel_pending_enqueues=True)
        sess.run(close_op)


def test(sess,
         queues,
         dbinterface,
         validation_targets,
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
    # Collect args in a dict of lists
    test_args = {
        'sess': sess,
        'queues': queues,
        'dbinterface': dbinterface,
        'validation_targets': validation_targets,
        'save_intermediate_freq': save_intermediate_freq}

    _ttargs = [{key: value[i] for (key, value) in test_args.items()}
               for i in range(len(sess))]

    for ttarg in _ttargs:

        ttarg['coord'], ttarg['threads'] = start_queues(ttarg['sess'])
        ttarg['dbinterface'].start_time_step = time.time()
        validation_summary = run_targets_dict(ttarg['sess'],
                                              ttarg['validation_targets'],
                                              save_intermediate_freq=ttarg['save_intermediate_freq'],
                                              dbinterface=ttarg['dbinterface'],
                                              validation_only=True)

    res = []
    for ttarg in _ttargs:
        ttarg['dbinterface'].sync_with_host()
        res.append(ttarg['dbinterface'].outrecs)
        stop_queues(ttarg['sess'], ttarg['queues'],
                    ttarg['coord'], ttarg['threads'])

    return validation_summary, res


def test_from_params(load_params,
                     model_params,
                     validation_params,
                     log_device_placement=False,
                     save_params=None,
                     dont_run=False,
                     skip_check=False,
                     inter_op_parallelism_threads=40,
                     ):
    """
    Main testing interface function.

    Same as train_from_parameters; but just performs testing without training.

    For documentation, see argument descriptions in train_from_params.

    """
    params, test_args = parse_params('test',
                                     model_params,
                                     dont_run=dont_run,
                                     skip_check=skip_check,
                                     save_params=save_params,
                                     load_params=load_params,
                                     validation_params=validation_params,
                                     log_device_placement=log_device_placement,
                                     inter_op_parallelism_threads=inter_op_parallelism_threads)

    with tf.Graph().as_default(), tf.device(DEFAULT_HOST):

        # create session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                log_device_placement=log_device_placement,
                                                inter_op_parallelism_threads=inter_op_parallelism_threads))

        init_op_global = tf.global_variables_initializer()
        sess.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        sess.run(init_op_local)
        log.info('Initialized from scratch first')

        # For convenience, use list of dicts instead of dict of lists
        _params = [{key: value[i] for (key, value) in params.items()}
                   for i in range(len(params['model_params']))]
        _ttargs = [{key: value[i] for (key, value) in test_args.items()}
                   for i in range(len(params['model_params']))]

        # Build a graph for each distinct model.
        for param, ttarg in zip(_params, _ttargs):
            # with tf.variable_scope(param['model_params']['prefix']) as scope:

            # print(scope.name)
            ttarg['dbinterface'] = DBInterface(params=param, load_params=param['load_params'])
            ttarg['dbinterface'].load_rec()
            ld = ttarg['dbinterface'].load_data
            assert ld is not None, "No load data found for query, aborting"
            ld = ld[0]
            # TODO: have option to reconstitute model_params entirely from
            # saved object ("revivification")
            param['model_params']['seed'] = ld['params']['model_params']['seed']
            cfg_final = ld['params']['model_params']['cfg_final']
            train_queue_params = ld['params']['train_params']['queue_params']

            (ttarg['validation_targets'],
             ttarg['queues']) = get_valid_targets_dict(loss_params=None,
                                                       cfg_final=cfg_final,
                                                       queue_params=train_queue_params,
                                                       **param)

            # tf.get_variable_scope().reuse_variables()

            param['load_params']['do_restore'] = True
            param['model_params']['cfg_final'] = cfg_final

            prefix = param['model_params']['prefix'] + '/'
            all_vars = variables._all_saveable_objects()
            var_list = strip_prefix(prefix, all_vars)

            ttarg['dbinterface'] = DBInterface(sess=sess,
                                               params=param,
                                               var_list=var_list,
                                               load_params=param['load_params'],
                                               save_params=param['save_params'])
            ttarg['dbinterface'].initialize(no_scratch=True)
            ttarg['sess'] = sess
            ttarg['save_intermediate_freq'] = param['save_params'].get('save_intermediate_freq')

        # Convert back to a dictionary of lists
        params = {key: [param[key] for param in _params]
                  for key in _params[0].keys()}
        test_args = {key: [ttarg[key] for ttarg in _ttargs]
                     for key in _ttargs[0].keys()}

        if dont_run:
            return test_args

        res = test(**test_args)
        sess.close()
        return res


def train_loop(sess, train_targets, num_minibatches=1):
    """Evaluate train_targets.

    Args:
        sess (tf.Session): Current tensorflow session.
        train_targets (dict): Training targets to be evaluated by sess.
        num_minibatches (int, optional): Number of minibatches to use.
            Defaults to 1.

    Returns:
        dict: The result of calling sess.run on the train_targets.
    """
    for minibatch in range(num_minibatches - 1):
        sess.run([target['grads'] for target in train_targets])
    return sess.run(train_targets)


def train(sess,
          queues,
          dbinterface,
          train_loop,
          train_targets,
          global_step,
          num_minibatches=1,
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
        - num_minibatches (int)
            How many minibatches to use to before applying gradient update.
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
        'num_minibatches': num_minibatches,
        'validation_targets': validation_targets}

    # Convert to a list of dicts
    trargs = [{key: value[i] for (key, value) in train_args.items()}
              for i in range(len(train_targets))]

    num_steps = [t['num_steps'] for t in trargs]
    steps = [t['global_step'].eval(session=t['sess']) for t in trargs]

    # Start queues and initial validation
    for (step, trarg) in zip(steps, trargs):

        if step >= trarg['num_steps']:
            log.info('Training cancelled since step ({}) is >= num_steps ({})'.
                     format(step, trarg['num_steps']))
            return

        log.info('Training beginning ...')
        trarg['coord'], trarg['threads'] = start_queues(trarg['sess'])

        if step == 0:
            trarg['dbinterface'].start_time_step = time.time()
            if trarg['validate_first']:
                valid_res = run_targets_dict(trarg['sess'],
                                             trarg['validation_targets'],
                                             dbinterface=trarg['dbinterface'])
    sess = train_args['sess'][0]
    train_loop = train_args['train_loop'][0]
    train_targets = train_args['train_targets']

    # Run training
    while any(step < num_step for (step, num_step) in zip(steps, num_steps)):

        # Make a single call to sess.run to produce a list of results.
        # TODO: call default train_loop which has minibatch option and is
        # defined in base.py

        start_time_step = time.time()
        if train_loop is not None:
            train_results = train_loop(sess,
                                       train_targets,
                                       num_minibatches=trarg['num_minibatches'])
        else:
            train_results = sess.run(train_targets)

        for (step, trarg, train_res) in zip(steps, trargs, train_results):

            old_step = step
            step = trarg['global_step'].eval(session=trarg['sess'])

            if step <= old_step:
                raise NoChangeError('Your optimizer should have incremented the global step,'
                                    ' but did not: old_step=%d, new_step=%d' % (old_step, step))
            if train_res['loss'] > trarg['thres_loss']:
                raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_res['loss'],
                                                                                     trarg['thres_loss']))

            # Validation
            vtargs = trarg['validation_targets'] if step % trarg['dbinterface'].save_valid_freq == 0 else {}
            valid_res = run_targets_dict(trarg['sess'], vtargs)

            # Save
            trarg['dbinterface'].start_time_step = start_time_step
            trarg['dbinterface'].save(train_res=train_res,
                                      valid_res=valid_res,
                                      validation_only=False)

        steps = [t['global_step'].eval(session=t['sess']) for t in trargs]

    # Sync and close the session
    res = []
    for trarg in trargs:
        stop_queues(trarg['sess'], trarg['queues'],
                    trarg['coord'], trarg['threads'])
        trarg['dbinterface'].sync_with_host()
        res.append(trarg['dbinterface'].outrecs)

    trargs[0]['sess'].close()
    return res


def train_from_params(save_params,
                      model_params,
                      train_params,
                      loss_params=None,
                      learning_rate_params=None,
                      optimizer_params=None,
                      validation_params=None,
                      log_device_placement=False,
                      load_params=None,
                      dont_run=False,
                      skip_check=False,
                      inter_op_parallelism_threads=40,
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
                        - "seed" -- seed for use in random generation of final config
                    - Must return:
                        - train output tensorflow nodes
                        - final configuration used in model
                - Remaining itmes in model_params are dictionary of arguments massed to func.

        - train_params (dict)
            Containing params for data sources and targets in training:
                - train_params['data'] contains params for the data
                    - train_params['data']['func'] is the function that constructs the data
                      provider.   This dataprovider must be an instance of a subclass of
                      tfutils.data.DataProviderBase.   Specifically, it must have a method
                      called "init_ops" -- see documentation in tfutils/data.py.
                    - remainder of train_params['data'] are kwargs passed to func
                - train_params['targets'] (optional) contains params for additional train targets
                    - train_params['targets']['func'] is a function that produces
                      tensorflow nodes as training targets
                    - remainder of train_parms['targets'] are arguments to func
                - train_params['queue_params'] is an optional dict of
                      params used to specify creation for the queue, passed to the
                      Queue.__init__ method.   Default is {}.
    :Kwargs:
        - loss_params (dict):
            Parameters for to utils.get_loss function for specifying loss
                - loss_params['targets'] is a string or a list of strings,
                  contain the names of inputs nodes that will be sent into the loss function
                  Must be provided
                - loss_params['loss_per_case_func'] is the function used to calculate the loss.
                  Must be provided.
                  The parameters sent to this function is defined by loss_params['loss_per_case_func_params'].
                - loss_params['loss_per_case_func_params'] is a dict including  help information about
                  how positional parameters should be sent to loss_params['loss_per_case_func'] as named parameters.
                  Default is {'_outputs': 'logits', '_targets_$all': 'labels'}
                    - If loss_params['loss_per_case_func_params'] is empty, the parameters for
                      loss_params['loss_per_case_func'] will be (outputs, *[inputs[t] for t in targets], **loss_func_kwargs),
                      where 'outputs' is the output of the network, inputs is the input nodes,
                      and targets is loss_params['targets'].
                    - Key value can have three choices:
                      - '_outputs': the value of this key will be the name for 'outputs'.
                      - '_targets_$all': name for '[inputs[t] for t in targets]'.
                      - '_target_somename': name for 'inputs[somename]' is somename is inside targets.
                    - Parameters not mentioned by the key values will still be sent to the function as positional parameters.
                - loss_params['agg_func'] is the aggregate function, default is None
                - loss_params['loss_func_kwargs']. Keyword parameters sent to loss_params['loss_per_case_func']. Default is None.
                - loss_params['agg_func_kwargs']. Keyword parameters sent to loss_params['agg_func']. Default is None.

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
                        'num_steps': (int) number of batches of validation source to compute
                        'agg_func': (optional, callable) how to aggregate validation results
                                across batches after computation. Signature is:
                                    - one input argument: the list of validation batch results
                                    - one output: aggregated version
                                Default is utils.identity_func
                        'online_agg_func': (optional, callable) how to aggregate validation results
                                on a per-batch basis. Siganture is:
                                    - three input arguments: (current aggregate, new result, step)
                                    - one output: new aggregated result
                                One first step, current aggregate passed in is None.
                                The final result is passed to the "agg_func".
                                Default is utils.append_and_return
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

        - thres_loss (float, default: 100)
            If loss exceeds this during training, HiLossError is thrown

        - num_steps (int or None, default: None)
            How many total steps of the optimization are run.  If None, train is run until process is cancelled.

      - load_params (dict)
            Dictionary of arguments for loading model, if different from saver
            (see Saver class)

        - log_device_placement (bool, default: False)
            Whether to log device placement in tensorflow session

        - inter_op_parallelism_threads (int, default: 40)
            Inter op thread pool size (has to be set large enough to avoid deadlock
            when using multiple queues)

    """
    params, train_args = parse_params('train',
                                      model_params,
                                      dont_run=dont_run,
                                      skip_check=skip_check,
                                      load_params=load_params,
                                      loss_params=loss_params,
                                      save_params=save_params,
                                      train_params=train_params,
                                      optimizer_params=optimizer_params,
                                      validation_params=validation_params,
                                      learning_rate_params=learning_rate_params,
                                      log_device_placement=log_device_placement,
                                      inter_op_parallelism_threads=inter_op_parallelism_threads)

    with tf.Graph().as_default(), tf.device(DEFAULT_HOST):

        # For convenience, use list of dicts instead of dict of lists
        _params = [{key: value[i] for (key, value) in params.items()}
                   for i in range(len(params['model_params']))]
        _trargs = [{key: value[i] for (key, value) in train_args.items()}
                   for i in range(len(params['model_params']))]

        # Use a single dataprovider for all models.
        data_params = _params[0]['train_params']['data_params']
        queue_params = _params[0]['train_params']['queue_params']

        (_params[0]['train_params']['data_params'],
         queues, inputs) = get_data(queue_params=queue_params, **data_params)

        # Build a graph for each distinct model.
        for param, trarg in zip(_params, _trargs):
            with tf.variable_scope(param['model_params']['prefix']):

                trarg['global_step'] = tf.get_variable('global_step', [],
                                                       dtype=tf.int64, trainable=False,
                                                       initializer=tf.constant_initializer(0))

                _, _, param, trarg = get_model(inputs,
                                               param['model_params'],
                                               param=param,
                                               trarg=trarg)

                tf.get_variable_scope().reuse_variables()

                (trarg['validation_targets'],
                 vqueue) = get_valid_targets_dict(queue_params=queue_params,
                                                  **param)
                queues.extend(vqueue)

        # Create session.

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                gpu_options=gpu_options,
                                                log_device_placement=log_device_placement,
                                                inter_op_parallelism_threads=inter_op_parallelism_threads))

        init_op_global = tf.global_variables_initializer()
        sess.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        sess.run(init_op_local)
        log.info('Initialized from scratch first')

        for param, trarg in zip(_params, _trargs):

            prefix = param['model_params']['prefix'] + '/'
            all_vars = variables._all_saveable_objects()
            var_list = strip_prefix(prefix, all_vars)

            trarg['dbinterface'] = DBInterface(sess=sess,
                                               params=param,
                                               var_list=var_list,
                                               global_step=trarg['global_step'],
                                               save_params=param['save_params'],
                                               load_params=param['load_params'])
            trarg['dbinterface'].initialize(no_scratch=True)
            trarg['sess'] = sess
            trarg['queues'] = queues

        # Convert back to a dictionary of lists
        params = {key: [param[key] for param in _params]
                  for key in _params[0].keys()}
        train_args = {key: [trarg[key] for trarg in _trargs]
                      for key in _trargs[0].keys()}

        if dont_run:
            return train_args

        return train(**train_args)


def get_valid_targets_dict(validation_params,
                           model_params,
                           loss_params,
                           queue_params,
                           cfg_final=None,
                           **params):
    """Helper function for creating validation target operations.

    NB: this function may modify validation_params.

    """
    valid_targets_dict = OrderedDict()
    queues = []
    model_params = copy.deepcopy(model_params)
    # model_params.pop('train', None)  # hackety-hack
    model_params['train'] = False
    prefix = model_params['prefix']
    if cfg_final is None:
        assert 'cfg_final' in model_params
        cfg_final = model_params['cfg_final']
    assert 'seed' in model_params
    for vtarg in validation_params:
        queue_params = validation_params[vtarg].get('queue_params', queue_params)
        _, queue, vinputs = get_data(queue_params=queue_params,
                                     **validation_params[vtarg]['data_params'])
        queues.extend(queue)
        # scope_name = 'validation/%s' % vtarg
        scope_name = '{}/validation/{}'.format(prefix, vtarg)
        with tf.name_scope(scope_name):
            _mp, voutputs = get_model(vinputs, model_params)
            check_model_equivalence(_mp['cfg_final'], cfg_final, scope_name)
            tf.get_variable_scope().reuse_variables()
        validation_params[vtarg], valid_targets_dict[vtarg] = get_validation_target(vinputs, voutputs,
                                                                                    **validation_params[vtarg])

    return valid_targets_dict, queues


def check_model_equivalence(m1, m2, name):
    """TODO: fill this in to make it stronger."""
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
    assert len(input_ops) == data_params[
        'n_threads'], (len(input_ops), data_params['n_threads'])
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

    temp_args = {v: tf.split(
        inputs[v], axis=0, num_or_size_splits=num_gpus) for v in inputs}
    list_of_args = [{now_arg: temp_args[now_arg][ind]
                     for now_arg in temp_args} for ind in xrange(num_gpus)]

    return list_of_args


def get_model_base(input, func, seed=0, train=False, **model_params):
    model_params['seed'] = seed
    model_params['train'] = train
    outputs, cfg_final = func(inputs=input,
                              **model_params)
    model_params['func'] = func
    model_params['cfg_final'] = cfg_final
    return model_params, outputs


def get_model(inputs, model_params, param=None, trarg=None):
    """Return model and any other targets (loss + optimizer) specified."""
    with tf.variable_scope(tf.get_variable_scope()):

        tower_outputs = []
        devices = model_params['devices']
        num_gpus = model_params['num_gpus']
        inputs = split_input(inputs, num_gpus)

        # DEFAULT: Prepare loss and optimizer if training
        if model_params['train']:
            assert param and trarg is not None

            tower_losses = []
            tower_grads = []

            (param['learning_rate_params'],
             learning_rate) = get_learning_rate(trarg['global_step'],
                                                **param['learning_rate_params'])
            (param['optimizer_params'],
             optimizer_base) = get_optimizer_base(learning_rate,
                                                  param['optimizer_params'])

        # Distribute graph across desired devices.
        for device, input in zip(devices, inputs):
            with tf.device(device), tf.name_scope('__GPU__' + device[-1]):

                model_params, output = get_model_base(input, **model_params)
                tower_outputs.append(output)

                tf.get_variable_scope().reuse_variables()

                # DEFAULT: Get loss and optimizer if training
                if model_params['train']:

                    (param['loss_params'],
                     loss) = get_loss(input, output, **param['loss_params'])

                    tf.get_variable_scope().reuse_variables()

                    grad = optimizer_base.compute_gradients(loss)
                    tower_losses.append(loss)
                    tower_grads.append(grad)

    # Gather outputs on the host (CPU)
    output = aggregate_outputs(tower_outputs)

    # DEFAULT: Accumulate and average gradients on the host (CPU)
    if model_params['train']:

        if param['train_params'].get('targets') is not None:
            ttargs = copy.deepcopy(param['train_params']['targets'])
            ttargs_func = ttargs.pop('func')
            ttarg = ttargs_func(input, output, **ttargs)
            trarg['train_targets'].update(ttarg)

        loss = tf.reduce_mean(tf.stack(tower_losses))
        minibatch_grads = optimizer_base.aggregate_gradients(tower_grads)
        grads = optimizer_base.accumulate_gradients(minibatch_grads, trarg['num_minibatches'])
        optimizer = optimizer_base.apply_gradients(grads, trarg['global_step'])
        zero_grad = optimizer_base.zero_grad()

        if 'loss' not in trarg['train_targets']:
            trarg['train_targets']['loss'] = loss
        if 'grads' not in trarg['train_targets']:
            trarg['train_targets']['grads'] = grads
        if 'optimizer' not in trarg['train_targets']:
            trarg['train_targets']['optimizer'] = optimizer
        if 'zero_grad' not in trarg['train_targets']:
            trarg['train_targets']['zero_grad'] = zero_grad
        if 'learning_rate' not in trarg['train_targets']:
            trarg['train_targets']['learning_rate'] = learning_rate

        param['model_params'] = model_params
        return param['model_params'], output, param, trarg
    else:
        return model_params, output


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


def parse_params(mode,
                 model_params,
                 dont_run=False,
                 skip_check=False,
                 save_params=None,
                 train_params=None,
                 loss_params=None,
                 load_params=None,
                 optimizer_params=None,
                 validation_params=None,
                 learning_rate_params=None,
                 log_device_placement=False,
                 inter_op_parallelism_threads=40,
                 ):
    """Ensure the params dictionary has the correct structure.

    Each *_params arg must be a list of dictionaries where the ith element
    corresponds to parameters of the ith distinct model. Thus, the length of
    all *_params must be the same and reflect the number of distinct models
    to be evaluated.

    If an *_params arg does not satisfy the above requirements, `parse_params`
    attempts to convert to the correct structure and logs any changes made.
    If it is missing any necessary components, defaults defined at the top of
    this module are used. If there exists an unresovlable conflict, an error
    is raised, and the user will be forced to resolve it before continuing.

    """
    model_params = [model_params] if not isinstance(model_params,
                                                    list) else model_params
    num_models = len(model_params)
    list_lens = [num_models]
    DEVICES = copy.copy(DEFAULT_DEVICES)

    params = {
        'dont_run': dont_run,
        'skip_check': skip_check,
        'load_params': load_params,
        'save_params': save_params,
        'model_params': model_params,
        'validation_params': validation_params,
        'log_device_placement': log_device_placement,
        'inter_op_parallelism_threads': inter_op_parallelism_threads}

    if mode == 'train':
        params.update({
            'loss_params': loss_params,
            'train_params': train_params,
            'optimizer_params': optimizer_params,
            'learning_rate_params': learning_rate_params})

    # Ensure params is a dict of lists, using defaults when necessary.
    for name, param_list in params.items():
        if not param_list and not isinstance(param_list, bool):
            param_list = DEFAULT_PARAMS[name]
        if not isinstance(param_list, list):
            param_list = [copy.deepcopy(param_list) for _ in range(num_models)]
        if len(param_list) != num_models and len(param_list) == 1:
            param_list += (num_models - 1) * copy.deepcopy(param_list)

        for model_num, param in enumerate(param_list):

            # Parse model_params.
            if name == 'model_params':
                if 'seed' not in param:
                    param['seed'] = DEFAULT_MODEL_SEED
                    log.info('No seed specified for model {}... '.format(model_num) +
                             'Defaulting to seed: {}.'.format(DEFAULT_MODEL_SEED))
                if 'prefix' not in param:
                    param['prefix'] = 'model_{}'.format(model_num)
                    log.info('No prefix specified for model {}... '.format(model_num) +
                             'Defaulting to prefix: {}.'.format(param['prefix']))
                if 'train' not in param:
                    if mode == 'train':
                        param['train'] = True
                    else:
                        param['train'] = False

                # Parse device specification.
                if 'devices' not in param:
                    param['devices'] = [DEVICES.pop(0)]
                    log.info('No devices specified for model {}... '.format(model_num) +
                             'Defaulting to gpus: {}.'.format(param['devices']))
                param['devices'] = format_devices(param['devices'])

                if 'num_gpus' not in param:
                    param['num_gpus'] = len(param['devices'])
                assert(param['num_gpus'] == len(param['devices']),
                       'num_gpus does not match the number of gpus specified in devices.')

            # Parse train_params.
            if name == 'train_params':
                if 'num_steps' not in param:
                    param['num_steps'] = DEFAULT_TRAIN_NUM_STEPS
                    log.info('num_steps not specified for model {}... '.format(model_num) +
                             'Defaulting num_steps to: {}.'.format(DEFAULT_TRAIN_NUM_STEPS))
                if 'thres_loss' not in param:
                    param['thres_loss'] = DEFAULT_TRAIN_THRES_LOSS
                    log.info('thres_loss not specified for model {}... '.format(model_num) +
                             'Defaulting thres_loss to: {}.'.format(DEFAULT_TRAIN_THRES_LOSS))
                if 'train_loop' not in param:
                    param['train_loop'] = {'func': train_loop}
                    log.info('train_loop not specified for model {}... '.format(model_num) +
                             'Using default training loop.')
                if 'validate_first' not in param:
                    param['validate_first'] = True
                    log.info('validate_fist not specified for model {}... '.format(model_num) +
                             'Defaulting validate_first to: {}.'.format(param['validate_first']))

                # Parse training data params (minibatching).
                if 'minibatch_size' not in param:
                    param['num_minibatches'] = 1
                    param['minibatch_size'] = param['data_params']['batch_size']
                    log.info('minibatch_size not specified for training data_params... ' +
                             'Defaulting minibatch_size to: {} (identical to the batch size).'
                             .format(param['data_params']['batch_size']))
                else:
                    batch_size = param['data_params']['batch_size']
                    minibatch_size = param['minibatch_size']
                    assert(minibatch_size <= batch_size,
                           'Minibatch size cannot be larger than batch size.')

                    num_minibatches = batch_size / float(minibatch_size)
                    if num_minibatches % 1 != 0:
                        num_minibatches = round(num_minibatches)
                        minibatch_size = batch_size / num_minibatches
                        log.warning('Minibatch size ({}) does not divide batch size ({}) evenly...'
                                    .format(minibatch_size, batch_size))
                        log.info('Rounding minibatch size to closest factor of batch size: {}'
                                 .format(minibatch_size))
                    param['minibatch_size'] = minibatch_size
                    param['num_minibatches'] = num_minibatches

        params[name] = param_list

        list_lens.append(len(param_list))
        assert isinstance(param_list, list), '{} should also be a list'.format(name)
        assert len(param_list) == num_models, '{} should have length'.format(num_models)
    assert len(np.unique(list_lens)) == 1, 'All param lists should have be same length!'

    # Append the model_prefix to non-unique exp_ids.
    for key in ['save_params', 'load_params']:
        unique_exp_ids = set(s.get('exp_id') for s in params[key])
        if None not in unique_exp_ids:
            if len(unique_exp_ids) == 1 and num_models != 1:
                log.warning('Non-unique exp_ids detected in {} '.format(key) +
                            'with multiple models.'.format(key))
                for i, (p, mp) in enumerate(zip(params[key],
                                                params['model_params'])):
                    p.update({'exp_id': p.get('exp_id') + '_' + mp['prefix']})
                    log.info('Appending \'_{} to the exp_id of model number {}.'.
                             format(mp['prefix'], i))
                    log.info('New exp_id is: {}'.format(p.get('exp_id')))

            assert len(set(s['exp_id'] for s in params[key])) == num_models

    # Prepare run_args to be passed to `base.(train|test)(**run_args)`.
    run_args = {
        'sess': num_models * [None],
        'queues': num_models * [None],
        'dbinterface': num_models * [None],
        'validation_targets': [dict() for _ in range(num_models)]}

    if mode == 'test':
        run_args.update({
            'save_intermediate_freq': num_models * [None]})
    else:
        run_args.update({
            'global_step': num_models * [None],
            'train_targets': [dict() for _ in range(num_models)],
            'num_steps': [p['num_steps'] for p in params['train_params']],
            'thres_loss': [p['thres_loss'] for p in params['train_params']],
            'train_loop': [p['train_loop']['func'] for p in params['train_params']],
            'validate_first': [p['validate_first'] for p in params['train_params']],
            'num_minibatches': [p['num_minibatches'] for p in params['train_params']]})

    return params, run_args


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
