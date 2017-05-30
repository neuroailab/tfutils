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


DEFAULT_LOAD_PARAMS = frozendict({'do_restore': True})


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
        recent_name = '_'.join([self.dbname, self.collname, self.exp_id, '__RECENT'])
        self.collfs_recent = gridfs.GridFS(self.conn[recent_name])

        self.load_data = None
        load_query = load_params.get('query')
        if load_query is None:
            load_query = {}
        else:
            if self.sameloc:
                raise Exception('Loading pointlessly')
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

    def initialize(self):
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
                log.info('Restored Vars:\n'+str([restore_var.name for restore_var in restore_vars]))
                tf_saver_restore = tf.train.Saver(restore_vars)
                # tensorflow restore
                log.info('Restoring variables from record %s (step %d)...' % (str(rec['_id']), rec['step']))
                tf_saver_restore.restore(self.sess, cache_filename)
                log.info('... done restoring.')
                all_variables = tf.global_variables() + tf.local_variables() # get list of all variables
                unrestored_vars = [var for var in all_variables \
                                            if var not in restore_vars] # compute list of variables not restored
                self.sess.run(tf.variables_initializer(unrestored_vars)) # initialize variables not restored
                assert len(self.sess.run(tf.report_uninitialized_variables())) == 0, self.sess.run(tf.report_uninitialized_variables())
        if not self.do_restore or self.load_data is None:
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
        results[target_name] = run_targets(sess,
                                           dbinterface,
                                           target_name,
                                           target,
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


def test_from_params(load_params,
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
        valid_targets_dict, queues = get_valid_targets_dict(validation_params,
                                                            model_params,
                                                            train_queue_params,
                                                            None,
                                                            cfg_final=cfg_final)
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
          queues,
          dbinterface,
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

    step = global_step.eval(session=sess)

    if step >= num_steps:
        log.info('Training cancelled since step (%d) is >= num_steps (%d)' % (step, num_steps))
        return

    log.info('Training beginning ...')
    coord, threads = start_queues(sess)

    if step == 0:
        dbinterface.start_time_step = time.time()
        if validate_first:
            validation_res = run_targets_dict(sess, validation_targets,
                                          dbinterface=dbinterface)
    while step < num_steps:
        old_step = step
        dbinterface.start_time_step = time.time()
        train_results = sess.run(train_targets)
        step = global_step.eval(session=sess)
        if step <= old_step:
            raise NoChangeError('Your optimizer should have incremented the global step,'
                                ' but did not: old_step=%d, new_step=%d' % (old_step, step))
        if train_results['loss'] > thres_loss:
            raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_results['loss'], thres_loss))

        vtargs = validation_targets if step % dbinterface.save_valid_freq == 0 else {}
        validation_res = run_targets_dict(sess, vtargs)
        dbinterface.save(train_res=train_results, valid_res=validation_res, validation_only=False)

    stop_queues(sess, queues, coord, threads)
    dbinterface.sync_with_host()
    return dbinterface.outrecs


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

    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]
        # TODO:  option to try to load first and see if records exist
        #        and if so, construct entirely from record ("revivification").
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      dtype=tf.int64,
                                      trainable=False)

        queue_params = train_params.get('queue_params')
        train_params['data_params'], train_inputs, queue = get_data(queue_params=queue_params,
                                                                    **train_params['data_params'])
        queues = [queue]

        if 'num_steps' not in train_params:
            train_params['num_steps'] = DEFAULT_TRAIN_NUM_STEPS
        if 'thres_loss' not in train_params:
            train_params['thres_loss'] = DEFAULT_TRAIN_THRES_LOSS

        model_params, train_outputs = get_model(train_inputs, train=True, **model_params)

        if loss_params is None:
            loss_params = {}
        loss_params, loss = get_loss(train_inputs, train_outputs, **loss_params)

        if learning_rate_params is None:
            learning_rate_params = {}
        learning_rate_params, learning_rate = get_learning_rate(global_step,
                                                                **learning_rate_params)

        optimizer_params, optimizer = get_optimizer(learning_rate,
                                                    loss,
                                                    global_step,
                                                    optimizer_params)

        if isinstance(loss, list):
            loss = tf.stack(loss)
            loss = tf.reduce_mean(loss)

        train_targets = {'loss': loss,
                         'learning_rate': learning_rate,
                         'optimizer': optimizer}
        if train_params.get('targets') is not None:
            ttargs_kwargs = copy.deepcopy(train_params['targets'])
            ttargs_func = ttargs_kwargs.pop('func')
            ttarg = ttargs_func(train_inputs, train_outputs, **ttargs_kwargs)
            train_targets.update(ttarg)

        scope = tf.get_variable_scope()
        scope.reuse_variables()
        if validation_params is None:
            validation_params = {}
        valid_targets_dict, vqueues = get_valid_targets_dict(validation_params,
                                                             model_params,
                                                             queue_params,
                                                             loss_params,
                                                             cfg_final=model_params['cfg_final'])
        queues.extend(vqueues)

        # create session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=log_device_placement,
            inter_op_parallelism_threads=inter_op_parallelism_threads))

        params = {'save_params': save_params,
                  'load_params': load_params,
                  'train_params': train_params,
                  'model_params': model_params,
                  'loss_params': loss_params,
                  'learning_rate_params': learning_rate_params,
                  'optimizer_params': optimizer_params,
                  'validation_params': validation_params,
                  'log_device_placement': log_device_placement,
                  'inter_op_parallelism_threads': inter_op_parallelism_threads}
        dbinterface = DBInterface(sess=sess, global_step=global_step, params=params,
                                  save_params=save_params, load_params=load_params)
        dbinterface.initialize()

        if dont_run:
            return sess, queues, dbinterface, train_targets, global_step, valid_targets_dict

        if 'validate_first' not in train_params:
            train_params['validate_first'] = True

        res = train(sess,
                    queues,
                    dbinterface,
                    train_targets=train_targets,
                    global_step=global_step,
                    num_steps=train_params['num_steps'],
                    thres_loss=train_params['thres_loss'],
                    validate_first=train_params['validate_first'],
                    validation_targets=valid_targets_dict)
        sess.close()
        return res


def get_valid_targets_dict(validation_params,
                           model_params,
                           default_queue_params,
                           default_loss_params,
                           cfg_final=None):
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
        _, vinputs, queue = get_data(queue_params=queue_params,
                                     **validation_params[vtarg]['data_params'])
        queues.append(queue)
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
                          agg_func=utils.identity_func,
                          online_agg_func=utils.append_and_return,
                          **validation_params):
    target_params = validation_params.get('targets', dict(default_target_params))
    func = target_params.pop('func', default_target_func)
    vtargets = func(vinputs, voutputs, **target_params)
    target_params['func'] = func
    validation_params['targets'] = target_params
    if 'num_steps' not in validation_params:
        assert hasattr(vinputs, 'total_batches'), '"num_batches" not specified in validation params, '\
            'data object must have "total_batches" attribute to be used as default.'
        validation_params['num_steps'] = vinputs.total_batches
    validation_params['agg_func'] = agg_func
    validation_params['online_agg_func'] = online_agg_func
    valid_targets = {'targets': vtargets,
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
    queue = get_queue(input_ops[0], **queue_params)
    for input_op in input_ops:
        if batch_size == 1:
            enqueue_ops.append(queue.enqueue(input_op))
        else:
            enqueue_ops.append(queue.enqueue_many(input_op))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue,
                                                                             enqueue_ops))
    inputs = queue.dequeue_many(queue_params['batch_size'])
    return data_params, inputs, queue


def get_model(train_inputs, func, seed=0, train=False, **model_params):
    model_params['seed'] = seed
    model_params['train'] = train
    outputs, cfg_final = func(inputs=train_inputs,
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


def get_learning_rate(global_step, func=tf.train.exponential_decay, **learning_rate_params):
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
