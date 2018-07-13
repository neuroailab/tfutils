"""base.py"""
from __future__ import absolute_import, division, print_function

import os
import re
import sys
import time
import importlib
import argparse
import json
import copy
import logging
import tarfile
import cPickle
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

import tfutils.utils as utils
from tfutils.data import get_queue
from tfutils.optimizer import ClipOptimizer
from tfutils.error import HiLossError, NoGlobalStepError, NoChangeError
from tfutils.utils import (sonify,
                   frozendict,
                   strip_prefix,
                   format_devices,
                   make_mongo_safe,
                   CoordinatedThread,
                   aggregate_outputs,
                   verify_pb2_v2_files,
                   get_saver_pb2_v2_files,
                   strip_prefix_from_name)


# tpu and estimator imports
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator
import pdb


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

DEFAULT_TPU_ZONE = None
DEFAULT_NUM_SHARDS = 8
DEFAULT_ITERATIONS_PER_LOOP = 100

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
DEFAULT_LOAD_PARAMS = frozendict({'do_restore': True, 'from_ckpt': None, 'to_restore': None, 'load_param_dict': None})
DEFAULT_LEARNING_RATE_PARAMS = frozendict({'func': tf.train.exponential_decay})

DEFAULT_LOSS_PARAMS = frozendict({'targets': ['labels'],
                                  'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
                                  'agg_func': tf.reduce_mean})

DEFAULT_OPTIMIZER_PARAMS = frozendict({'optimizer_class': tf.train.MomentumOptimizer,
                                       'momentum': 0.9})

DEFAULT_SAVE_PARAMS = frozendict({'save_metrics_freq': 100,
                                  'save_valid_freq': 3000,
                                  'cache_max_num': 6,
                                  'cache_filters_freq': 3000,
                                  'save_filters_freq': 30000,
                                  'save_initial_filters': True,
                                  'save_to_gfs': (),
                                  'do_save': True})

DEFAULT_PARAMS = frozendict({
    'dont_run': False,
    'skip_check': False,
    'model_params': frozendict(),
    'train_params': frozendict(),
    'validation_params': frozendict(),
    'log_device_placement': False,
    'inter_op_parallelism_threads': 40,
    'save_params': frozendict(DEFAULT_SAVE_PARAMS),
    'load_params': frozendict(DEFAULT_LOAD_PARAMS),
    'loss_params': frozendict(DEFAULT_LOSS_PARAMS),
    'optimizer_params': frozendict(DEFAULT_OPTIMIZER_PARAMS),
    'learning_rate_params': frozendict(DEFAULT_LEARNING_RATE_PARAMS),
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
                    - cache_max_num (int, default: 6)
                        Maximal number of cached filters to keep in __RECENT database
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
                    - from_ckpt (string)
                        Path to load from a TensorFlow checkpoint (instead of from the db)
                    - to_restore (list of strings or a regex/callable which returns strings)
                        Specifies which variables should be loaded from the checkpoint.
                        Any variables not specified here will be reinitialized.
                    - load_param_dict (dict)
                        A dictionary whose keys are the names of the variables that are to be loaded
                        from the checkpoint, and the values are the names of the variables of the model
                        that you want to restore with the value of the corresponding checkpoint variable.
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
            self.sameloc = self.sameloc & (load_params['query']['exp_id'] == self.exp_id)

        for _k in ['do_save', 'save_metrics_freq', 'save_valid_freq', 'cache_filters_freq', 'cache_max_num',
                   'save_filters_freq', 'save_initial_filters', 'save_to_gfs']:
            setattr(self, _k, save_params.get(_k, DEFAULT_SAVE_PARAMS[_k]))

        for _k in ['do_restore', 'from_ckpt', 'to_restore', 'load_param_dict']:
            setattr(self, _k, load_params.get(_k, DEFAULT_LOAD_PARAMS[_k]))

        self.rec_to_save = None
        self.checkpoint_thread = None
        self.outrecs = []

        self.conn = pymongo.MongoClient(host=self.host, port=self.port)
        self.conn.server_info()
        self.collfs = gridfs.GridFS(self.conn[self.dbname], self.collname)

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

        if (save_params == {}) and ('cache_dir' in load_params): # use cache_dir from load params if save_params not given
            cache_dir = load_params['cache_dir']
        elif 'cache_dir' in save_params:
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
                raise Exception('You specified load parameters but no '
                                'record was found with the given spec.')
        self.load_data = load

    def initialize(self, no_scratch=False):
        """Fetch record then uses tf's saver.restore."""
        if self.do_restore:

            # First, determine which checkpoint to use.
            if self.from_ckpt is not None:
                # Use a cached checkpoint file.
                ckpt_filename = self.from_ckpt
                log.info('Restoring variables from checkpoint %s ...' % ckpt_filename)
            else:
                # Otherwise, use a database checkpoint.
                self.load_rec() if self.load_data is None else None
                if self.load_data is not None:
                    rec, ckpt_filename = self.load_data
                    log.info('Restoring variables from record %s (step %d)...' %
                             (str(rec['_id']), rec['step']))
                else:
                    # No db checkpoint to load.
                    ckpt_filename = None

            if ckpt_filename is not None:

                all_vars = tf.global_variables() + tf.local_variables()  # get list of all variables
                self.all_vars = strip_prefix(self.params['model_params']['prefix'], all_vars)

                # Next, determine which vars should be restored from the specified checkpoint.
                restore_vars = self.get_restore_vars(ckpt_filename, self.all_vars)
                restore_stripped = strip_prefix(self.params['model_params']['prefix'], list(restore_vars.values()))
                restore_names =  [name for name, var in restore_stripped.items()]
                # Actually load the vars.
                log.info('Restored Vars:\n' + str(restore_names))
                tf_saver_restore = tf.train.Saver(restore_vars)
                tf_saver_restore.restore(self.sess, ckpt_filename)
                log.info('... done restoring.')

                # Reinitialize all other, unrestored vars.
                unrestored_vars = [var for name, var in self.all_vars.items() if name not in restore_names]
                unrestored_var_names = [name for name, var in self.all_vars.items() if name not in restore_names]
                log.info('Unrestored Vars:\n' + str(unrestored_var_names))
                self.sess.run(tf.variables_initializer(unrestored_vars))  # initialize variables not restored
                assert len(self.sess.run(tf.report_uninitialized_variables())) == 0, (
                    self.sess.run(tf.report_uninitialized_variables()))

        if not self.do_restore or (self.load_data is None and self.from_ckpt is None):
            init_op_global = tf.global_variables_initializer()
            self.sess.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            self.sess.run(init_op_local)

    def get_restore_vars(self, save_file, all_vars=None):
        """Create the `var_list` init argument to tf.Saver from save_file.

        Extracts the subset of variables from tf.global_variables that match the
        name and shape of variables saved in the checkpoint file, and returns these
        as a list of variables to restore.

        To support multi-model training, a model prefix is prepended to all
        tf global_variable names, although this prefix is stripped from
        all variables before they are saved to a checkpoint. Thus,


        Args:
            save_file: path of tf.train.Saver checkpoint.

        Returns:
            dict: checkpoint variables.

        """
        reader = tf.train.NewCheckpointReader(save_file)
        var_shapes = reader.get_variable_to_shape_map()
        log.info('Saved Vars:\n' + str(var_shapes.keys()))

        var_shapes = {  # Strip the prefix off saved var names.
            strip_prefix_from_name(self.params['model_params']['prefix'], name): shape
            for name, shape in var_shapes.items()}

        # Map old vars from checkpoint to new vars via load_param_dict.
        mapped_var_shapes = self.remap_var_list(var_shapes)
        log.info('Saved shapes:\n' + str(mapped_var_shapes))

        if all_vars is None:
            all_vars = tf.global_variables() + tf.local_variables()  # get list of all variables
            all_vars = strip_prefix(self.params['model_params']['prefix'], all_vars)

        # Specify which vars are to be restored vs. reinitialized.
        if self.load_param_dict is None:
            restore_vars = {name: var for name, var in all_vars.items() if name in mapped_var_shapes}
        else:
            # associate checkpoint names with actual variables
            load_var_dict = {}
            for ckpt_var_name, curr_var_name in self.load_param_dict.items():
                for curr_name, curr_var in all_vars.items():
                    if curr_name == curr_var_name:
                        load_var_dict[ckpt_var_name] = curr_var
                        break

            restore_vars = load_var_dict

        restore_vars = self.filter_var_list(restore_vars)

        # Ensure the vars to restored have the correct shape.
        var_list = {}
        for name, var in restore_vars.items():
            var_shape = var.get_shape().as_list()
            if var_shape == mapped_var_shapes[name]:
                var_list[name] = var
        return var_list

    def remap_var_list(self, var_list):
        """Map old vars in checkpoint to new vars in current session.

        Args:
            var_list (dict): var names mapped to variables (or some related
            quantity, such as variable shapes).

        Returns:
            dict: New var names mapped to the corresponding restored var.

        Examples:
        >>>var_list
        {'Weights': <tf.Variable>}
        >>>self.load_param_dict
        {'Weights': 'Filters'}
        >>>self.remap_var_list(var_list)
        {'Filters': <tf.Variable>}

        """
        if self.load_param_dict is None:
            log.info('No variable mapping specified.')
            return var_list
        for old_name, new_name in self.load_param_dict.items():
            for name in var_list:
                if old_name == name:
                    var_list[old_name] = var_list.pop(old_name)
                    break
        return var_list

    def filter_var_list(self, var_list):
        """Filter checkpoint vars for those to be restored.

        Args:
            checkpoint_vars (list): Vars that can be restored from checkpoint.
            to_restore (list[str] or regex, optional): Selects vars to restore.

        Returns:
            list: Variables to be restored from checkpoint.

        """
        if not self.to_restore:
            return var_list
        elif isinstance(self.to_restore, re._pattern_type):
            return {name: var for name, var in var_list.items()
                    if self.to_restore.match(name)}
        elif isinstance(self.to_restore, list):
            return {name: var for name, var in var_list.items()
                    if name in self.to_restore}
        raise TypeError('to_restore ({}) unsupported.'.format(type(self.to_restore)))

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
            raise er.OperationFailure(inst.args[0] + "\n Is your dbname too long? Mongo requires that dbnames be no longer than 64 characters.")
        if count_recent > 0:  # get latest that matches query
            ckpt_record_recent = coll_recent.find(query, sort=[('uploadDate', -1)])[0]
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
                fsbucket = gridfs.GridFSBucket(database, bucket_name=loading_from.name.split('.')[0])
                fsbucket.download_to_stream(ckpt_record['_id'], load_dest)
                load_dest.close()
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
            # TODO: also include error rate of the train set to monitor overfitting
            message = 'Step {} ({:.0f} ms) -- '.format(step, 1000 * duration)
            msg2 = ['{}: {:.4f}'.format(k, v) for k, v in train_res.items()
                    if k not in ['optimizer', '__grads__'] and k not in self.save_to_gfs]
            message += ', '.join(msg2)
            log.info(message)

            if '__grads__' in train_res:
                del train_res['__grads__']
            if 'optimizer' in train_res:
                del train_res['optimizer']
            if 'train_results' not in rec:
                rec['train_results'] = []
            rec['train_results'].append(train_res)

        # print validation set performance
        if len(valid_res) > 0:
            rec['validation_results'] = valid_res
            message = 'Validation -- '
            message += ', '.join('{}: {}'.format(
                k, {_k: _v for _k, _v in v.items()
                if _k not in self.save_to_gfs}) for k, v in valid_res.items())
            log.info(message)

        if validation_only:
            if self.sess is not None:
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

            save_rec = sonify(rec, skip=self._skip_check)
            make_mongo_safe(save_rec)

            coord = tf.train.Coordinator()
            thread = CoordinatedThread(coord=coord,
                                       target=self._save_thread,
                                       args=(save_filters_permanent,
                                             save_filters_tmp,
                                             save_rec,
                                             step,
                                             save_to_gfs))
            thread.daemon = True
            thread.start()
            self.checkpoint_thread = thread
            self.checkpoint_coord = coord

    def sync_with_host(self):
        if self.checkpoint_thread is not None:
            try:
                self.checkpoint_coord.join([self.checkpoint_thread])
            except Exception as error:
                log.warning('A checkpoint thead raised an exception '
                            'while saving a checkpoint.')
                log.error(error)
                raise
            else:
                self.checkpoint_thread = None

    def _save_thread(self, save_filters_permanent, save_filters_tmp, save_rec, step, save_to_gfs):
        if save_filters_permanent or save_filters_tmp:
            save_rec['saved_filters'] = True
            save_path = os.path.join(self.cache_dir, 'checkpoint')
            log.info('Saving model with path prefix %s ... ' % save_path)
            if self.sess is not None: # can be None if using estimator
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
                recent_gridfs_files = self.collfs_recent._GridFS__files
                recent_query_result = recent_gridfs_files.find({'saved_filters': True}, sort=[('uploadDate', 1)])
                num_cached_filters = recent_query_result.count()
                cache_max_num = self.cache_max_num
                if num_cached_filters > cache_max_num:
                    log.info('Cleaning up cached filters')
                    fsbucket = gridfs.GridFSBucket(recent_gridfs_files._Collection__database, bucket_name=recent_gridfs_files.name.split('.')[0])

                    for del_indx in xrange(0, num_cached_filters - cache_max_num):
                        #log.info(recent_query_result[del_indx]['uploadDate'])
                        fsbucket.delete(recent_query_result[del_indx]['_id'])

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

    Args:
        sess (tensorflow.Session): Object in which to run calculations
        queues (list of CustomQueue): Objects containing asynchronously queued data iterators
        dbinterface (DBInterface object): Saver through which to save results
        validation_targets (dict of tensorflow objects): Objects on which validation will be computed.
        save_intermediate_freq (None or int): How frequently to save intermediate results captured during test
            None means no intermediate saving will be saved

    Returns:
        dict: Validation summary.
        dict: Results.

    """
    # Collect args in a dict of lists
    test_args = {
        'queues': queues,
        'dbinterface': dbinterface,
        'validation_targets': validation_targets,
        'save_intermediate_freq': save_intermediate_freq}

    _ttargs = [{key: value[i] for (key, value) in test_args.items()}
               for i in range(len(queues))]

    for ttarg in _ttargs:

        ttarg['coord'], ttarg['threads'] = start_queues(sess)
        ttarg['dbinterface'].start_time_step = time.time()
        validation_summary = run_targets_dict(sess,
                                              ttarg['validation_targets'],
                                              save_intermediate_freq=ttarg['save_intermediate_freq'],
                                              dbinterface=ttarg['dbinterface'],
                                              validation_only=True)

    res = []
    for ttarg in _ttargs:
        ttarg['dbinterface'].sync_with_host()
        res.append(ttarg['dbinterface'].outrecs)
        stop_queues(sess, ttarg['queues'], ttarg['coord'], ttarg['threads'])

    return validation_summary, res


def test_from_params(load_params,
                     model_params,
                     validation_params,
                     log_device_placement=False,
                     save_params=None,
                     dont_run=False,
                     skip_check=False,
                     inter_op_parallelism_threads=40,
                     use_estimator=False
                     ):
    """
    Main testing interface function.

    Same as train_from_parameters; but just performs testing without training.

    For documentation, see argument descriptions in train_from_params.

    CAUTION: for tpu and estimators, test_from_params is UNtested.
    """
    # use tpu only if a tpu_name has been specified
    use_tpu = (model_params.get('tpu_name', None) is not None)
    if use_tpu:
        log.info('Using tpu: %s' %model_params['tpu_name'])

    params, test_args = parse_params('test',
                                      model_params,
                                      dont_run=dont_run,
                                      skip_check=skip_check,
                                      save_params=save_params,
                                      load_params=load_params,
                                      validation_params=validation_params,
                                      log_device_placement=log_device_placement,
                                      inter_op_parallelism_threads=inter_op_parallelism_threads,
                                      use_tpu=use_tpu)


    
    # do not need to create sess with estimator interface
    if use_estimator or use_tpu:
        # use this for tpu and estimator logging
        tf.logging.set_verbosity(tf.logging.INFO)
        # For convenience, use list of dicts instead of dict of lists
        _params = [{key: value[i] for (key, value) in params.items()}
                   for i in range(len(params['model_params']))]
        _ttargs = [{key: value[i] for (key, value) in test_args.items()}
                   for i in range(len(params['model_params']))]

        param = _params[0]
        ttarg = _ttargs[0]
        # Support only single model
        assert(len(_params) == 1)

        model_params = param['model_params']
        validation_params = param['validation_params']
        save_params = param['save_params']

        # store a dictionary of estimators, one for each validation params target
        # since may have a different set of eval steps to run on tpu
        # if dict of estimators not feasible, can just create one single estimator
        # and run its predict method multiple times on the same data function in test_estimator (I think)
        cls_dict = {}
        for valid_k in validation_params.keys():
            # set up estimator func
            valid_target_parameter = validation_params[valid_k]
            estimator_fn, params_to_pass = create_test_estimator_fn(use_tpu=use_tpu, 
                                               model_params=model_params,
                                               target_params=valid_target_parameter)
            validation_data_params = valid_target_parameter['data_params']
            eval_val_steps = valid_target_parameter['num_steps']

            if use_tpu:
                # grab tpu name and gcp, etc from model params
                m_config = create_test_tpu_config(model_dir=save_params.get('cache_dir', ''),
                                             eval_steps=eval_val_steps,
                                             tpu_name=model_params.get('tpu_name', None), 
                                             gcp_project=model_params.get('gcp_project', None), 
                                             tpu_zone=model_params.get('tpu_zone', DEFAULT_TPU_ZONE), 
                                             num_shards=model_params.get('num_shards', DEFAULT_NUM_SHARDS),
                                             iterations_per_loop=model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP))

                estimator_classifier = tpu_estimator.TPUEstimator(
                                            use_tpu=True,
                                            model_fn=estimator_fn,
                                            config=m_config,
                                            train_batch_size=validation_data_params['batch_size'],
                                            predict_batch_size=validation_data_params['batch_size'],
                                            params=params_to_pass)

            else:
                # TO DO need to verify non tpu estimator support
                estimator_classifier = tf.estimator.Estimator(model_fn=estimator_fn, params=params_to_pass)

            cls_dict[valid_k] = estimator_classifier
        return test_estimator(cls_dict=cls_dict, param=param, ttarg=ttarg)

    else:
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

                if not 'cache_dir' in load_params:
                    temp_cache_dir = save_params.get('cache_dir', None)
                    load_params['cache_dir'] = temp_cache_dir
                    log.info('cache_dir not found in load_params, using cache_dir ({}) from save_params'.format(temp_cache_dir))

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
                ttarg['save_intermediate_freq'] = param['save_params'].get('save_intermediate_freq')

            # Convert back to a dictionary of lists
            params = {key: [param[key] for param in _params]
                      for key in _params[0].keys()}
            test_args = {key: [ttarg[key] for ttarg in _ttargs]
                         for key in _ttargs[0].keys()}

            if dont_run:
                return test_args

            res = test(sess, **test_args)
            sess.close()
            return res


def train_loop(sess, train_targets, num_minibatches=1, **loop_params):
    """Define default minibatch training loop.

    A training loop that performs minibatching with ``num_minibatches``
    minibatches.

    Args:
        sess (tf.Session): Current tensorflow session.
        train_targets (dict): Target operations to be evaluated by ``sess.run``.
            By default, ``base.train_from_params`` inserts the following
            targets to facilitate minibatching:
            * ``__grads__`` (tf.Operation): Accumulates and stores gradients.
            * ``optimizer`` (tf.Operation): Applies and zeros gradients.
        num_minibatches (int): number of minibatches to use.
        **loop_params (mapping): additional, user-defined kwargs to
            be used in the training loop.

    Returns:
        dict: A dictionary containing train targets evaluated by the session.

    """
    assert all([required in targets for targets in train_targets
                for required in ['__grads__', 'optimizer']])

    # Perform minibatching
    range_len = (int)(num_minibatches)
    for minibatch in range(range_len - 1):
        # Accumulate gradient for each minibatch
        sess.run([target['__grads__'] for target in train_targets])

    # Compute final targets (includes zeroing gradient accumulator variable)

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
    """Actually runs the training evaluation loop.

    Args:
        sess (tesorflow.Session):
            Object in which to run calculations.
        queues (list of Queue): Objects containing asynchronously queued data iterators.

        dbinterface (DBInterface object): Saver through which to save results.

        train_loop (callable withs args: sess and train_targets):
            Callable that specifies a custom training loop
        train_targets (dict of tensorflow nodes): Targets to train.
            One item in this dict must be "optimizer" or similar
            to make anything happen
        num_minibatches (int): How many minibatches to use to before applying gradient update.
        num_steps (int): How many steps to train to before quitting
        validation_targets (dict of tensorflow objects, default: None):
            Objects on which validation will be computed
        thres_loss (float, default: 100):
            If loss exceeds this during training, HiLossError is thrown

    """
    # Collect args in a dict of lists
    train_args = {
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
    steps = [t['global_step'].eval(session=sess) for t in trargs]

    # Start queues and initial validation
    for (step, trarg) in zip(steps, trargs):

        if step >= trarg['num_steps']:
            log.info('Training cancelled since step ({}) is >= num_steps ({})'.
                     format(step, trarg['num_steps']))
            return

        log.info('Training beginning ...')
        trarg['coord'], trarg['threads'] = start_queues(sess)

        if step == 0:
            trarg['dbinterface'].start_time_step = time.time()
            if trarg['validate_first']:
                valid_res = run_targets_dict(sess,
                                             trarg['validation_targets'],
                                             dbinterface=trarg['dbinterface'])
    train_loop = train_args['train_loop'][0]
    train_targets = train_args['train_targets']

    # Run training
    while any(step < num_step for (step, num_step) in zip(steps, num_steps)):

        start_time_step = time.time()
        train_results = train_loop(sess, train_targets, num_minibatches=trarg['num_minibatches'])

        for (step, trarg, train_res) in zip(steps, trargs, train_results):

            old_step = step
            step = trarg['global_step'].eval(session=sess)

            if step <= old_step:
                raise NoChangeError('Your optimizer should have incremented the global step,'
                                    ' but did not: old_step=%d, new_step=%d' % (old_step, step))
            if train_res['loss'] > trarg['thres_loss']:
                raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_res['loss'],
                                                                                     trarg['thres_loss']))

            # Validation
            vtargs = trarg['validation_targets'] if step % trarg['dbinterface'].save_valid_freq == 0 else {}
            valid_res = run_targets_dict(sess, vtargs)

            # Save
            trarg['dbinterface'].start_time_step = start_time_step
            trarg['dbinterface'].save(train_res=train_res,
                                      valid_res=valid_res,
                                      validation_only=False)

        steps = [t['global_step'].eval(session=sess) for t in trargs]

    # Sync and close the session
    res = []
    for trarg in trargs:
        stop_queues(sess, trarg['queues'], trarg['coord'], trarg['threads'])
        trarg['dbinterface'].sync_with_host()
        res.append(trarg['dbinterface'].outrecs)

    sess.close()
    return res

def train_estimator(cls,
                    param,
                    trarg):

    model_dir = param['save_params'].get('cache_dir', '')
    train_steps = param['train_params']['num_steps']
    # only single targets during eval mode
    need_val = len(param['validation_params'].keys())>0
    steps_per_eval = param['save_params'].get('save_valid_freq')
    if need_val:
        valid_k = param['validation_params'].keys()[0]
        validation_data_params = param['validation_params'][valid_k]['data_params']
        valid_steps = param['validation_params'][valid_k]['num_steps']
        valid_fn = validation_data_params['func']
        if steps_per_eval is None:
            steps_per_eval = param['save_params']['save_filters_freq']
        else:
            save_filters_freq = param['save_params'].get('save_filters_freq')
            if save_filters_freq is not None:
                # these need to be the same right now because estimator loads
                # from last checkpoint after validating
                assert(steps_per_eval == save_filters_freq)
            else:
                param['save_params']['save_filters_freq'] = steps_per_eval
    train_fn = param['train_params']['data_params']['func'] 

    model_params = param['model_params']
    iterations_per_loop = model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP)

    if (steps_per_eval is None) or (steps_per_eval < iterations_per_loop): # eval steps cannot be less than TPU iterations
        log.info('Setting save_valid_freq ({}) to be the same as iterations_per_loop ({}).'.format(steps_per_eval, iterations_per_loop))
        steps_per_eval = iterations_per_loop

    train_hooks = param['train_params'].get('hooks')
    if need_val:
        valid_hooks = param['validation_params'][valid_k].get('hooks')
    else:
        valid_hooks = None

    current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)
    # initialize db here (currently no support for loading and saving to different places. May need to modify init so load_params can load from different dir, estimator interface limited
    #    when loading and saving to different paths, may need to create a new config)

    trarg['dbinterface'] = DBInterface(sess=None,
                                   params=param,
                                   global_step=current_step,
                                   save_params=param['save_params'],
                                   load_params=param['load_params'],
                                   cache_dir=model_dir)


    log.info('Training beginning ...')
    log.info('Training for %d steps. Current '
                    'step %d' % (train_steps,
                                 current_step))

    trarg['dbinterface'].start_time_step = time.time()

    tpu_validate_first = param['train_params'].get('tpu_validate_first', False)
    def do_tpu_validation():
        log.info('Starting to evaluate.')
        eval_results = cls.evaluate(
          input_fn=valid_fn,
          hooks=valid_hooks,
          steps=valid_steps)
        log.info('Saving eval results to database.')
        trarg['dbinterface'].save(valid_res={valid_k: eval_results}, validation_only=True)
        log.info('Done saving eval results to database.')

        return eval_results

    if tpu_validate_first:
        eval_results = do_tpu_validation()

    while current_step < train_steps:
        next_eval = min(current_step + steps_per_eval,
                            train_steps)

        log.info('Training until step %d' % next_eval)
        cls.train(
        input_fn=train_fn, max_steps=next_eval, hooks=train_hooks)
        current_step = next_eval

        if need_val:
            eval_results = do_tpu_validation()
    
    # sync with hosts
    res = []
    trarg['dbinterface'].sync_with_host()
    res.append(trarg['dbinterface'].outrecs)
    # returning final eval results for convenience
    return eval_results, res

def test_estimator(cls_dict,
                    param,
                    ttarg):

    # load params query stores path to checkpoint
    if param['load_params']['do_restore'] and (param['load_params']['query'] is not None):
        # path to specific checkpoint
        load_dir = param['load_params']['query']
    else:
        # gets latest checkpoint from model_dir
        load_dir = None

    ttarg['dbinterface'] = DBInterface(sess=None,
                                   params=param,
                                   save_params=param['save_params'],
                                   load_params=param['load_params'])


    ttarg['dbinterface'].start_time_step = time.time()

    m_predictions = {}
    for valid_k in cls_dict.keys():
        cls = cls_dict[valid_k]
        validation_data_params = param['validation_params'][valid_k]['data_params']
        # can use to filter particular params to save, if not there will set to None and all saved
        filter_keys = param['validation_params'][valid_k].get('keys_to_save') 
        session_hooks = param['validation_params'][valid_k].get('hooks') 
        valid_fn = validation_data_params['func']
        log.info('Starting to evaluate ({}).'.format(valid_k))
        eval_results = cls.predict(
          input_fn=valid_fn,
          predict_keys=filter_keys,
          hooks=session_hooks,
          checkpoint_path=load_dir)
        m_predictions[valid_k] = list(eval_results)

    log.info('Saving eval results to database.')
    # set validation only to be True to just save the results and not filters
    ttarg['dbinterface'].save(valid_res=m_predictions, validation_only=True)
    log.info('Done saving eval results to database.')
    
    # sync with hosts
    res = []
    ttarg['dbinterface'].sync_with_host()
    res.append(trarg['dbinterface'].outrecs)
    # returning final eval results for convenience
    return eval_results, res

def create_train_estimator_fn(use_tpu, 
                        model_params,
                        lr_params,
                        opt_params,
                        loss_params,
                        validation_params):

    """
    Creates a model_fn for use with tf.Estimator for training and eval.
    """
    # set up loss params 
    loss_agg_func = loss_params.get('agg_func', DEFAULT_LOSS_PARAMS['agg_func'])

    loss_per_case_func = loss_params.get('loss_per_case_func', DEFAULT_LOSS_PARAMS['loss_per_case_func'])

    loss_func_kwargs = loss_params.get('loss_func_kwargs', {})

    loss_agg_func_kwargs = loss_params.get('agg_func_kwargs', {})

    # tells clip optimizer to use tpu
    if opt_params['func'].__name__ == 'ClipOptimizer':
        opt_params['use_tpu'] = use_tpu

    # build params dictionary to be instantiated with the model_fn
    params_to_pass = {}
    params_to_pass['model_params'] = model_params
    params_to_pass['opt_params'] = opt_params
    params_to_pass['loss_agg_func'] = loss_agg_func
    params_to_pass['loss_per_case_func'] = loss_per_case_func
    params_to_pass['loss_func_kwargs'] = loss_func_kwargs
    params_to_pass['loss_agg_func_kwargs'] = loss_agg_func_kwargs
    params_to_pass['lr_params'] = lr_params

    def model_fn(features, labels, mode, params):
        model_params = params['model_params']
        opt_params = params['opt_params']
        loss_agg_func = params['loss_agg_func']
        loss_per_case_func = params['loss_per_case_func']
        loss_func_kwargs = params['loss_func_kwargs']
        loss_agg_func_kwargs = params['loss_agg_func_kwargs']
        lr_params = params['lr_params']

        model_params['train'] = (mode==tf.estimator.ModeKeys.TRAIN)
        if opt_params['use_tpu']:
            model_params['batch_size'] = params['batch_size'] # per shard batch_size

        model_func = model_params.pop('func')

        outputs = model_func(inputs=features, **model_params)
        if isinstance(outputs, dict):
            logit_key = model_params.get('logit_key', 'logits')
            logits = outputs[logit_key]
        else:
            logits = outputs
            
        loss_args = (outputs, labels)
        loss = loss_per_case_func(*loss_args, **loss_func_kwargs)
        loss = loss_agg_func(loss, **loss_agg_func_kwargs)

        global_step = tf.train.get_global_step()

        lr_func = lr_params.pop('func')
        learning_rate = lr_func(global_step=global_step, **lr_params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            opt_func = opt_params.pop('func', ClipOptimizer)
            optimizer_base = opt_func(learning_rate=learning_rate, **opt_params)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer_base.minimize(loss, global_step)
        else:
            train_op = None


        eval_metrics = None
        if mode == tf.estimator.ModeKeys.EVAL:
           num_valid_targets = len(validation_params.keys())
           metric_fn_kwargs = {'labels': labels, 'logits': logits}
           if opt_params['use_tpu']:
               assert(num_valid_targets==1) # tpu estimators currently only support single targets :(
               first_valid = validation_params.keys()[0]
               valid_target = validation_params[first_valid]['targets']
               metric_fn = valid_target['func']
               if isinstance(outputs, dict):
                   for kw in outputs.keys():
                       if kw != logit_key:
                           kw_val = outputs[kw]
                           new_kw = kw
                           if isinstance(new_kw, int):
                               new_kw = 'i%i' % new_kw
                           metric_fn_kwargs.update({new_kw:kw_val})

               for kw in valid_target.keys():
                   v = valid_target[kw]
                   if isinstance(v, dict):
                       for kw1 in v.keys():
                           # add any additional kwargs
                           kw_val = v[kw1]
                           metric_fn_kwargs.update({kw1: kw_val})
                           #metric_fn_kwargs[kw] = kw_val
               eval_metrics = (metric_fn, metric_fn_kwargs)
           else:
               # normal estimators expect dicts and can support multiple targets (but same dataset and eval_steps etc)
               eval_dict = {}
               for k in validation_params.keys():
                   k_metric_fn_kwargs = metric_fn_kwargs
                   k_target = k['targets']
                   for kw in k_target.keys():
                       if kw != 'func':
                           # add any additional kwargs
                           kw_val = k_target[kw]
                           k_metric_fn_kwargs[kw] = kw_val 
                           # k_metric_fn_kwargs.update(kw_val)                         
                   eval_dict[k] = (k_target['func'], k_metric_fn_kwargs)
               eval_metrics = eval_dict

        if opt_params['use_tpu']:
            return tpu_estimator.TPUEstimatorSpec(
              mode=mode,
              loss=loss,
              train_op=train_op,
              eval_metrics=eval_metrics)
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metrics)

    return model_fn, params_to_pass

def create_test_estimator_fn(use_tpu, 
                        model_params,
                        target_params):

    """
    Creates a model_fn for use with tf.Estimator for eval only. Uses predict mode.
    """
    # build params dictionary to be instantiated with the model_fn
    params_to_pass = {}
    params_to_pass['model_params'] = model_params
    params_to_pass['target_params'] = target_params
    params_to_pass['use_tpu'] = use_tpu

    def model_fn(features, labels, mode, params):
        model_params = params['model_params']
        target_params = params['target_params']
        model_params['train'] = (mode==tf.estimator.ModeKeys.TRAIN)
        if params['use_tpu']:
            model_params['batch_size'] = params['batch_size'] # per shard batch_size

        model_func = model_params.pop('func')
        outputs = model_func(inputs=features, **model_params)
        if isinstance(outputs, dict):
            logit_key = model_params.get('logit_key')
            if logit_key is None:
                logit_key = 'logits'
            logits = outputs[logit_key]
        else:
            logits = outputs

        predictions = None
        if mode == tf.estimator.ModeKeys.PREDICT:
           metric_fn_kwargs = {'labels': labels, 'logits': logits}

           eval_dict = {}
           k_metric_fn_kwargs = metric_fn_kwargs
           k_target = target_params['targets']
           for kw in k_target.keys():
               if kw != 'func':
                   # add any additional kwargs
                   kw_val = k_target[kw]
                   k_metric_fn_kwargs[kw] = kw_val 
                   # k_metric_fn_kwargs.update(kw_val)  
           k_target_func = k_target.pop('func')                       
           eval_dict['predictions'] = k_target_func(**k_metric_fn_kwargs)
           predictions = eval_dict

        if params['use_tpu']:
            return tpu_estimator.TPUEstimatorSpec(
              mode=mode,
              predictions=predictions)
        else:
            return tf.Estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

    return model_fn, params_to_pass

def create_train_tpu_config(model_dir,
                      model_params,
                      tpu_name,
                      gcp_project,
                      steps_per_checkpoint,
                      tpu_zone=DEFAULT_TPU_ZONE,
                      num_shards=DEFAULT_NUM_SHARDS,
                      keep_checkpoint_max=5,
                      iterations_per_loop=DEFAULT_ITERATIONS_PER_LOOP):

    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=[tpu_name],
            zone=tpu_zone,
            project=gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    if iterations_per_loop == -1 or (steps_per_checkpoint is not None and steps_per_checkpoint < iterations_per_loop):
        log.info('Setting iterations_per_loop ({}) to be the same as steps_per_checkpoint ({}).'.format(iterations_per_loop, steps_per_checkpoint))
        iterations_per_loop = steps_per_checkpoint
        model_params['iterations_per_loop'] = iterations_per_loop

    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=model_dir,
        save_checkpoints_steps=steps_per_checkpoint,
        save_checkpoints_secs=None,
        keep_checkpoint_max=keep_checkpoint_max,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_shards))

    return config

def create_test_tpu_config(model_dir,
                      eval_steps,
                      tpu_name,
                      gcp_project,
                      tpu_zone=DEFAULT_TPU_ZONE,
                      num_shards=DEFAULT_NUM_SHARDS,
                      iterations_per_loop=DEFAULT_ITERATIONS_PER_LOOP):

    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=[tpu_name],
            zone=tpu_zone,
            project=gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=model_dir,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=eval_steps,
            num_shards=num_shards))

    return config

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
                      use_estimator=False
                      ):
    """
    Main training interface function.

    Args:
        use_estimator (bool): Whether to use tf.Estimator interface to train and evaluate models (default False).
            In this case, the session is created internally. Will automatically be used when using tpus. 

        save_params (dict): Dictionary of arguments for creating saver object (see Saver class).

        model_params (dict): Containing function that produces model and arguments to that function.
            model_params['func'] is the function producing the model.

            The function's signature is:
            ::

                inputs: data object
                - ``train`` -- boolean if training is happening
                - ``seed`` -- seed for use in random generation of final config

            Returns:
            (tf.Operations): train output tensorflow nodes
            - final configuration used in model
            - Remaining items in model_params are dictionary of arguments massed to func.

        train_params (dict): Containing params for data sources and targets in training.

            - ``train_params['data']`` contains params for the data

            - ``train_params['data']['func']`` is the function that constructs the data
              provider. This dataprovider must be an instance of a subclass of
              tfutils.data.DataProviderBase. Specifically, it must have a method
              called "init_ops" -- see documentation in tfutils/data.py.

            - Remainder of ``train_params['data']`` are kwargs passed to func.

            - ``train_params['targets']`` (optional) contains params for additional train targets

            - ``train_params['targets']['func']`` is a function that produces
              tensorflow nodes as training targets

            - Remainder of ``train_parms['targets']`` are arguments to func.

            - ``train_params['queue_params']`` is an optional dict of
              params used to specify creation for the queue, passed to the
              Queue.__init__ method.   Default is {}.

        loss_params (dict): Parameters for to utils.get_loss function for specifying loss.

            - ``loss_params['targets']` is a string or a list of strings,
              contain the names of inputs nodes that will be sent into the loss function

            - ``loss_params['loss_per_case_func']`` is the function used to calculate the loss.
              Must be provided. The parameters sent to this function is defined by loss_params['loss_per_case_func_params'].

            - ``loss_params['loss_per_case_func_params']`` is a dict including  help information about
              how positional parameters should be sent to loss_params['loss_per_case_func'] as named parameters.
              Default is ``{'_outputs': 'logits', '_targets_': 'labels'}``

            - If ``loss_params['loss_per_case_func_params']`` is empty, the parameters for
              loss_params['loss_per_case_func'] will be (outputs, *[inputs[t] for t in targets], **loss_func_kwargs),
              where 'outputs' is the output of the network, inputs is the input nodes,
              and targets is ``loss_params['targets']``.

            Key value can have three choices:
            - '_outputs': the value of this key will be the name for 'outputs'.
            - '_targets_': name for '[inputs[t] for t in targets]'.
            - '_target_somename': name for 'inputs[somename]' is somename is inside targets.

        - Parameters not mentioned by the key values will still be sent to the function as positional parameters.
            - ``loss_params['agg_func']`` is the aggregate function, default is None
            - ``loss_params['loss_func_kwargs']``. Keyword parameters sent to loss_params['loss_per_case_func']. Default is None.
            - ``loss_params['agg_func_kwargs']`. Keyword parameters sent to ``loss_params['agg_func']. Default is None.

        learning_rate_params (dict): Parameters for specifying learning_rate.
                - :obj:`learning_rate_params['func']` is a function producing
                  tensorflow node acting as learning rate. This function must accept argument "global_step".
                - remainder of learning_rate_params are arguments to func.

        optimizer_params (dict): Parameters for creating optimizer.
            - optimizer_params['func'] is a function producing a
              tensorflow optimizer object (like a subclass of tf.train.Optimizer)

            Must accept:
            - "learning_rate" -- the result of the learning_rate_func call
            - Must return object with a method called "minimize" with
              the same call signature as tensorflow.train.Optimizer.minimize --- that is:
            - Must accept:
                * "loss" -- result of loss_func call
                * "global_step" -- global step used for determine learning rate,
            Must return:
                * tensorflow node which computes gradients and applies
                  them, and must increment "global_step"
            - Remainder of optimizer_params (aside form "func") are arguments
              to the optimizer func

        validation_params (dict): Dictionary of validation sources. The structure if this dictionary is:

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

        queue_params (dict, defualt: None): Dictionary of arguments to Queue object (see
            tfutils.data.Queue documentation)

        thres_loss (float, default: 100): If loss exceeds this during training, HiLossError is thrown

        num_steps (int or None, default: None): How many total steps of the optimization are run.
            If None, train is run until process is cancelled.

        load_params (dict): Dictionary of arguments for loading model, if different from saver
            (see Saver class).

        log_device_placement (bool, default: False): Whether to log device placement in tensorflow session

        inter_op_parallelism_threads (int, default: 40): Inter op thread pool size (has to be set large enough to avoid deadlock
            when using multiple queues)

    Deleted Parameters:
        saver_params

    Returns:
        TYPE: Description.

    """
    # use tpu only if a tpu_name has been specified
    use_tpu = (model_params.get('tpu_name', None) is not None)
    if use_tpu:
        log.info('Using tpu: %s' %model_params['tpu_name'])
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
                                      inter_op_parallelism_threads=inter_op_parallelism_threads,
                                      use_tpu=use_tpu or use_estimator)


    
    # do not need to create sess with estimator interface
    if use_estimator or use_tpu:
        # use this for tpu and estimator logging
        tf.logging.set_verbosity(tf.logging.INFO)
        # For convenience, use list of dicts instead of dict of lists
        _params = [{key: value[i] for (key, value) in params.items()}
                   for i in range(len(params['model_params']))]
        _trargs = [{key: value[i] for (key, value) in train_args.items()}
                   for i in range(len(params['model_params']))]

        param = _params[0]
        trarg = _trargs[0]
        # Support only single model
        assert(len(_params) == 1)
        train_data_params = param['train_params']['data_params']

        model_params = param['model_params']
        lr_params = param['learning_rate_params']
        opt_params = param['optimizer_params']
        loss_params = param['loss_params']
        validation_params = param['validation_params']
        save_params = param['save_params']
        # set up estimator func
        estimator_fn, params_to_pass = create_train_estimator_fn(use_tpu=use_tpu, 
                                           model_params=model_params,
                                           lr_params=lr_params,
                                           opt_params=opt_params,
                                           loss_params=loss_params,
                                           validation_params=validation_params)

        if use_tpu:
            if len(param['validation_params'].keys())>0:
                valid_k = param['validation_params'].keys()[0]
                validation_data_params = param['validation_params'][valid_k]['data_params']
                eval_batch_size = validation_data_params['batch_size']
            else:
                eval_batch_size = None
            # grab tpu name and gcp, etc from model params
            m_config = create_train_tpu_config(model_dir=save_params.get('cache_dir', ''),
                                         tpu_name=model_params.get('tpu_name', None), 
                                         gcp_project=model_params.get('gcp_project', None), 
                                         steps_per_checkpoint=save_params.get('save_filters_freq', None),
                                         tpu_zone=model_params.get('tpu_zone', DEFAULT_TPU_ZONE), 
                                         num_shards=model_params.get('num_shards', DEFAULT_NUM_SHARDS),
                                         keep_checkpoint_max=save_params.get('checkpoint_max', 5),
                                         iterations_per_loop=model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP),
                                         model_params=model_params)

            estimator_classifier = tpu_estimator.TPUEstimator(
                                        use_tpu=True,
                                        model_fn=estimator_fn,
                                        config=m_config,
                                        train_batch_size=train_data_params['batch_size'],
                                        eval_batch_size=eval_batch_size,
                                        params=params_to_pass)

        else:
            # TO DO need to verify non tpu estimator support, may be good to use RunConfig here
            estimator_classifier = tf.estimator.Estimator(model_fn=estimator_fn, params=params_to_pass)

        return train_estimator(cls=estimator_classifier, param=param, trarg=trarg)

    else:
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
                for var in var_list:
                    print(var)

                trarg['dbinterface'] = DBInterface(sess=sess,
                                                   params=param,
                                                   var_list=var_list,
                                                   global_step=trarg['global_step'],
                                                   save_params=param['save_params'],
                                                   load_params=param['load_params'])
                trarg['dbinterface'].initialize()
                trarg['queues'] = queues

            # Convert back to a dictionary of lists
            params = {key: [param[key] for param in _params]
                      for key in _params[0].keys()}
            train_args = {key: [trarg[key] for trarg in _trargs]
                          for key in _trargs[0].keys()}

            if dont_run:
                return train_args

            return train(sess, **train_args)


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
                     'num_steps': validation_params['num_steps']}
    return validation_params, valid_targets


def get_data(func, queue_params=None, **data_params):
    data_provider = func(**data_params)
    input_ops = data_provider.init_ops()
    assert len(input_ops) == data_params['n_threads'], (len(input_ops), data_params['n_threads'])
    assert len(input_ops) > 0, len(input_ops)
    batch_size = data_params['batch_size']
    data_params['func'] = func
    enqueue_ops = []
    queue = get_queue(input_ops[0], shape_flag=batch_size!=1, **queue_params)
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
    if not isinstance(num_gpus, list):
        n_gpus = num_gpus
    else:
        n_gpus = len(num_gpus)

    if n_gpus == 1:
        return [inputs]

    temp_args = {v: tf.split(inputs[v], axis=0, num_or_size_splits=num_gpus)
                 for v in inputs}

    list_of_args = [{now_arg: temp_args[now_arg][ind]
                     for now_arg in temp_args} for ind in xrange(n_gpus)]

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
    """Return model and any other targets (loss + optimizer) specified.

    Args:
        inputs (tf.Operation): Model inputs provided by a tf.QueueRunner.
        model_params (dict): Specifies model configuration and must contain:
            'devices' (list): device specs (e.g. '/gpu:0')
            'train' (bool): whether getting model for training
        param (None, optional): Description.
        trarg (None, optional): Description.
        inputs ()

    Returns:
        tuple: Description.

    """
    with tf.variable_scope(tf.get_variable_scope()):

        tower_outputs = []
        devices = model_params['devices']
        num_gpus = model_params['num_gpus']
        inputs = split_input(inputs, num_gpus)
        # DEFAULT: Prepare loss and optimizer if training.
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

    # Gather and aggregate outputs on the host (CPU).
    output = aggregate_outputs(tower_outputs)

    # DEFAULT: Accumulate and average gradients on the host (CPU).
    if model_params['train']:

        if param['train_params'].get('targets') is not None:
            ttargs = copy.deepcopy(param['train_params']['targets'])
            ttargs_func = ttargs.pop('func')
            ttarg = ttargs_func(input, output, **ttargs)
            trarg['train_targets'].update(ttarg)

        # Aggregate loss.
        loss = tf.reduce_mean(tf.stack(tower_losses))

        # Aggregate and accumulate gradients.
        minibatch_grads = optimizer_base.aggregate_gradients(tower_grads)
        mini_flag, grads = optimizer_base.accumulate_gradients(minibatch_grads, trarg['num_minibatches'])
        #grads = minibatch_grads

        # Apply accumulated gradients.
        optimizer = optimizer_base.apply_gradients(grads, trarg['global_step'])

        # Prepare train_targets
        if 'loss' not in trarg['train_targets']:
            trarg['train_targets']['loss'] = loss
        if '__grads__' not in trarg['train_targets']:
            trarg['train_targets']['__grads__'] = mini_flag
            pass
        if 'optimizer' not in trarg['train_targets']:
            trarg['train_targets']['optimizer'] = optimizer
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
                 use_tpu=False
                 ):
    """Ensure the params dictionary has the correct structure.

    Each params arg must be a list of dictionaries where the ith element
    corresponds to parameters of the ith distinct model. Thus, the length of
    all params must be the same and reflect the number of distinct models
    to be evaluated.

    If an params arg does not satisfy the above requirements, ``parse_params``
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
            if isinstance(DEFAULT_PARAMS[name], frozendict):
                param_list = dict(DEFAULT_PARAMS[name])
            else:
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

                if not use_tpu:
                    # Parse device specification.
                    if 'devices' not in param:
                        param['devices'] = [DEVICES.pop(0)]
                        log.info('No devices specified for model {}... '.format(model_num) +
                                 'Defaulting to gpus: {}.'.format(param['devices']))
                    param['devices'] = format_devices(param['devices'])

                    if 'num_gpus' not in param:
                        param['num_gpus'] = len(param['devices'])

                    if not isinstance(param['num_gpus'], list):
                        assert param['num_gpus'] == len(param['devices']), (
                           'num_gpus does not match the number of gpus specified in devices.')
                    else:
                        assert len(param['num_gpus']) == len(param['devices']), (
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
                    if not use_tpu:
                        param['minibatch_size'] = param['queue_params']['batch_size']
                        log.info('minibatch_size not specified for training data_params... ' +
                                 'Defaulting minibatch_size to: {} (identical to the batch size).'
                                 .format(param['queue_params']['batch_size']))
                else:
                    if use_tpu:
                        batch_size = param['train_params']['batch_size']
                        minibatch_size = param['minibatch_size']
                        if batch_size != minibatch_size:
                            log.info('Minibatching currently not supported on TPU. Setting minibatch size ({}) ' +
                                     'to be original batch size ({}).'.format(minibatch_size, batch_size))
                        param['minibatch_size'] = batch_size
                        params['num_minibatches'] = 1
                    else:
                        batch_size = param['queue_params']['batch_size']
                        minibatch_size = param['minibatch_size']
                        assert minibatch_size <= batch_size, (
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
                        param['queue_params']['batch_size'] = minibatch_size

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
