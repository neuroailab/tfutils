import pymongo
from pymongo import errors as er
import gridfs
import tarfile
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
from bson.objectid import ObjectId
import datetime
from tensorflow.python import DType
import numpy as np
import collections
import inspect
import os
import tensorflow as tf
import copy
import time
from tensorflow.core.protobuf import saver_pb2
import re
import sys
import threading
import git
import pdb
import pkg_resources
from six import string_types

from tfutils.utils import strip_prefix_from_name, \
        strip_prefix, get_var_list_wo_prefix
from tfutils.helper import log
from tfutils.defaults import DEFAULT_SAVE_PARAMS, DEFAULT_LOAD_PARAMS

if 'TFUTILS_HOME' in os.environ:
    TFUTILS_HOME = os.environ['TFUTILS_HOME']
else:
    TFUTILS_HOME = os.path.join(os.environ['HOME'], '.tfutils')


def version_info(module):
    """Get version of a standard python module.

    Args:
        module (module): python module object to get version info for.

    Returns:
        dict: dictionary of version info.

    """
    if hasattr(module, '__version__'):
        version = module.__version__
    elif hasattr(module, 'VERSION'):
        version = module.VERSION
    else:
        pkgname = module.__name__.split('.')[0]
        try:
            info = pkg_resources.get_distribution(pkgname)
        except (pkg_resources.DistributionNotFound, pkg_resources.RequirementParseError):
            version = None
            log.warning(
                'version information not found for %s -- what package is this from?' % module.__name__)
        else:
            version = info.version

    return {'version': version}


def git_info(repo):
    """Return information about a git repo.

    Args:
        repo (git.Repo): The git repo to be investigated.

    Returns:
        dict: Git repo information

    """
    if repo.is_dirty():
        log.warning('repo %s is dirty -- having committment issues?' %
                    repo.git_dir)
        clean = False
    else:
        clean = True
    branchname = repo.active_branch.name
    commit = repo.active_branch.commit.hexsha
    origin = repo.remote('origin')
    urls = map(str, list(origin.urls))
    remote_ref = [_r for _r in origin.refs if _r.name ==
                  'origin/' + branchname]
    if not len(remote_ref) > 0:
        log.warning('Active branch %s not in origin ref' % branchname)
        active_branch_in_origin = False
        commit_in_log = False
    else:
        active_branch_in_origin = True
        remote_ref = remote_ref[0]
        gitlog = remote_ref.log()
        shas = [_r.oldhexsha for _r in gitlog] + \
            [_r.newhexsha for _r in gitlog]
        if commit not in shas:
            log.warning('Commit %s not in remote origin log for branch %s' % (commit,
                                                                              branchname))
            commit_in_log = False
        else:
            commit_in_log = True
    info = {'git_dir': repo.git_dir,
            'active_branch': branchname,
            'commit': commit,
            'remote_urls': urls,
            'clean': clean,
            'active_branch_in_origin': active_branch_in_origin,
            'commit_in_log': commit_in_log}
    return info


def verify_pb2_v2_files(cache_prefix, ckpt_record):
    file_data = get_saver_pb2_v2_files(cache_prefix)
    ndf = file_data['num_data_files']
    sndf = ckpt_record['_saver_num_data_files']
    assert ndf == sndf, (ndf, sndf)


def get_saver_pb2_v2_files(prefix):
    dirn, pref = os.path.split(prefix)
    pref = pref + '.'
    ldirn = os.listdir(dirn)
    files = list(filter(lambda x: x.startswith(pref) and not x.endswith('.tar'),
                   ldirn))
    indexf = pref + 'index'
    assert indexf in files, (prefix, indexf, files)
    notindexfiles = [_f for _f in files if _f != indexf]
    p = re.compile(pref + 'data-([\d]+)-of-([\d]+)$')
    total0 = None
    fns = []
    for f in notindexfiles:
        match = p.match(f)
        assert match, (f, prefix)
        thisf, total = map(int, match.groups())
        if total0 is None:
            total0 = total
        else:
            assert total == total0, (f, total, total0)
        fns.append(thisf)
    fns = list(set(fns))
    fns.sort()
    assert fns == list(range(total0)), (fns, total0)
    files = [os.path.join(dirn, f) for f in files]
    file_data = {'files': files, 'num_data_files': total0}
    return file_data


def make_mongo_safe(_d):
    """Make a json-izable actually safe for insertion into Mongo.

    Args:
        _d (dict): a dictionary to make safe for Mongo.

    """
    klist = list(_d.keys())
    for _k in klist:
        if hasattr(_d[_k], 'keys'):
            make_mongo_safe(_d[_k])
        if not isinstance(_k, str):
            _d[str(_k)] = _d.pop(_k)
        _k = str(_k)
        if '.' in _k:
            _d[_k.replace('.', '___')] = _d.pop(_k)


def version_check_and_info(module):
    """Return either git info or standard module version if not a git repo.

    Args:
        module (module): python module object to get info for.

    Returns:
        dict: dictionary of info

    """
    srcpath = inspect.getsourcefile(module)
    try:
        repo = git.Repo(srcpath, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        log.info('module %s not in a git repo, checking package version' %
                 module.__name__)
        info = version_info(module)
    else:
        info = git_info(repo)
    info['source_path'] = srcpath
    return info


def sonify(arg, memo=None, skip=False):
    """Return version of arg that can be trivally serialized to json format.

    Args:
        arg (object): an argument to sonify.
        memo (dict, optional): A dictionary to contain args. Defaults to None.
        skip (bool, optional): Skip git repo info check. Defaults to False.

    Returns:
        Sonified return argument.

    Raises:
        TypeError: Cannot sonify argument type.

    """
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]

    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, datetime.datetime):
        rval = arg
    elif isinstance(arg, DType):
        rval = arg
    elif isinstance(arg, np.floating):
        rval = float(arg)
    elif isinstance(arg, np.integer):
        rval = int(arg)
    elif isinstance(arg, (list, tuple)):
        rval = type(arg)([sonify(ai, memo, skip) for ai in arg])
    elif isinstance(arg, collections.OrderedDict):
        rval = collections.OrderedDict([(sonify(k, memo, skip),
                                         sonify(v, memo, skip))
                                        for k, v in arg.items()])
    elif isinstance(arg, dict):
        rval = dict([(sonify(k, memo, skip), sonify(v, memo, skip))
                     for k, v in arg.items()])
    elif isinstance(arg, (string_types, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = sonify(arg.sum(), skip=skip)
        else:
            rval = map(sonify, arg)  # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    elif callable(arg):
        mod = inspect.getmodule(arg)
        modname = mod.__name__
        objname = arg.__name__
        if not skip:
            rval = version_check_and_info(mod)
        else:
            rval = {}
        rval.update({'objname': objname,
                     'modname': modname})
        rval = sonify(rval, skip=skip)
    else:
        raise TypeError('sonify', arg)

    memo[id(rval)] = rval
    return rval


class DBInterface(object):

    def __init__(self,
                 var_manager=None,
                 params=None,
                 save_params=None,
                 load_params=None,
                 sess=None,
                 global_step=None,
                 cache_dir=None,
                 tfsaver_args=[],
                 tfsaver_kwargs={}):
        """
        :Kwargs:
            - params (dict)
                Describing all parameters of experiment
            - save_params (dict)
            - load_params (dict)
            - sess (tesorflow.Session)
                Object in which to run calculations.  This is required if actual loading/
                saving is going to be done (as opposed to just e.g. getting elements from
                the MongoDB).
            - global_step (tensorflow.Variable)
                Global step variable, the one that is updated by apply_gradients.  This
                is required if being using in a training context.
            - tfsaver_args, tsaver_kwargs
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
        self.var_manager = var_manager
        if self.var_manager:
            self.var_list = get_var_list_wo_prefix(params, var_manager)
        else:
            all_vars = tf.global_variables()
            self.var_list = {v.op.name: v for v in all_vars}

        # Set save_params and load_params:
        #   And set these parameters as attributes in this instance
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

        # Determine whether this loading is from the same location as saving
        self.sameloc = all([getattr(self, _k) == getattr(
            self, 'load_' + _k) for _k in location_variables])
        if 'query' in load_params \
                and not load_params['query'] is None \
                and 'exp_id' in load_params['query']:
            self.sameloc = \
                    self.sameloc \
                    & (load_params['query']['exp_id'] == self.exp_id)

        # Set some attributes only in save_params
        for _k in [\
                'do_save', 'save_metrics_freq', \
                'save_valid_freq', 'cache_filters_freq', 'cache_max_num', \
                'save_filters_freq', 'save_initial_filters', 'save_to_gfs']:
            setattr(self, _k, save_params.get(_k, DEFAULT_SAVE_PARAMS[_k]))

        # Set some attributes only in load_params
        for _k in ['do_restore', 'from_ckpt', 'to_restore', 'load_param_dict']:
            setattr(self, _k, load_params.get(_k, DEFAULT_LOAD_PARAMS[_k]))

        self.rec_to_save = None
        self.checkpoint_thread = None
        self.outrecs = []

        # Set the save mongo client
        self.conn = pymongo.MongoClient(host=self.host, port=self.port)
        self.conn.server_info()
        self.collfs = gridfs.GridFS(self.conn[self.dbname], self.collname)

        # Set the cache mongo client
        recent_name = '_'.join(
                [self.dbname, self.collname, self.exp_id, '__RECENT'])
        self.collfs_recent = gridfs.GridFS(self.conn[recent_name])

        self.load_data = None
        load_query = load_params.get('query')
        if load_query is None:
            load_query = {}
        else:
            # Special situation here
            # Users try to load from the same place they try to save through
            # setting the load_query
            # This is not allowed
            if self.sameloc and (not save_params == {}):
                raise Exception(
                        'Loading pointlessly! '\
                        + 'If you want to continue your training, '\
                        + 'please set your load_query to be None!')
            else:
                self.sameloc = False
        if 'exp_id' not in load_query:
            load_query.update({'exp_id': self.load_exp_id})
        self.load_query = load_query

        # Set the load mongo client
        if self.load_host != self.host or self.port != self.load_port:
            self.load_conn = pymongo.MongoClient(host=self.load_host,
                                                 port=self.load_port)
            self.load_conn.server_info()
        else:
            self.load_conn = self.conn
        self.load_collfs = gridfs.GridFS(self.load_conn[self.load_dbname],
                                         self.load_collname)
        # Set the cache mongo client for loading
        load_recent_name = '_'.join([self.load_dbname,
                                     self.load_collname,
                                     self.load_exp_id,
                                     '__RECENT'])
        self.load_collfs_recent = gridfs.GridFS(
            self.load_conn[load_recent_name])

        # Set the cache_dir: where to put local cache files
        # use cache_dir from load params if save_params not given
        if (save_params == {}) and ('cache_dir' in load_params):
            cache_dir = load_params['cache_dir']
        elif 'cache_dir' in save_params:
            cache_dir = save_params['cache_dir']
        else:
            cache_dir = None

        if not cache_dir:
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

    def initialize(self):
        """Fetch record then uses tf's saver.restore."""
        if self.do_restore:
            # First, determine which checkpoint to use.
            if self.from_ckpt is not None:
                # Use a cached checkpoint file.
                ckpt_filename = self.from_ckpt
                log.info('Restoring variables from checkpoint %s ...' \
                        % ckpt_filename)
            else:
                # Otherwise, use a database checkpoint.
                self.load_rec() if self.load_data is None else None
                if self.load_data is not None:
                    rec, ckpt_filename = self.load_data
                    log.info('Restoring variables from record %s (step %d)...' \
                             % (str(rec['_id']), rec['step']))
                else:
                    # No db checkpoint to load.
                    ckpt_filename = None

            if ckpt_filename is not None:
                # Determine which vars should be restored from the specified checkpoint.
                restore_vars = self.get_restore_vars(ckpt_filename)
                restore_names = [name for name, var in restore_vars.items()]
                # remap the actually restored names
                if self.load_param_dict:
                    new_restore_names = []
                    for each_old_name in restore_names:
                        new_restore_names.append(
                                self.load_param_dict[each_old_name])
                    restore_names = new_restore_names

                # Actually load the vars.
                log.info('Restored Vars (in ckpt, in graph):\n'
                         str(restore_names))
                tf_saver_restore = tf.train.Saver(restore_vars)
                tf_saver_restore.restore(self.sess, ckpt_filename)
                log.info('... done restoring.')

                # Run post init_ops if needed
                if self.var_manager:
                    self.sess.run(
                            tf.group(*self.var_manager.get_post_init_ops()))

                # Reinitialize all other, unrestored vars.
                unrestored_vars = [\
                        var \
                        for name, var in self.var_list.items() \
                        if name not in restore_names]
                unrestored_var_names = [\
                        name \
                        for name, var in self.var_list.items() \
                        if name not in restore_names]
                log.info('Unrestored Vars (in graph, not in ckpt):\n'
                         str(unrestored_var_names))
                self.sess.run(tf.variables_initializer(unrestored_vars))  # initialize variables not restored
                assert len(self.sess.run(tf.report_uninitialized_variables())) == 0, (
                    self.sess.run(tf.report_uninitialized_variables()))

        if not self.do_restore \
                or (self.load_data is None and self.from_ckpt is None):
            init_op_global = tf.global_variables_initializer()
            self.sess.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            self.sess.run(init_op_local)
            if self.var_manager:
                self.sess.run(tf.group(*self.var_manager.get_post_init_ops()))

    def get_restore_vars(self, save_file):
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

        # Map old vars from checkpoint to new vars via load_param_dict.
        log.info('Saved vars and shapes:\n' + str(var_shapes))

        # Specify which vars are to be restored vs. reinitialized.
        all_vars = self.var_list
        if not self.load_param_dict:
            restore_vars = {
                    name: var for name, var in all_vars.items() \
                            if name in var_shapes}
        else:
            # associate checkpoint names with actual variables
            load_var_dict = {}
            for ckpt_var_name, curr_var_name in self.load_param_dict.items():
                if curr_var_name in all_vars:
                    load_var_dict[ckpt_var_name] = all_vars[curr_var_name]
            restore_vars = load_var_dict

        restore_vars = self.filter_var_list(restore_vars)

        # These variables are stored in the checkpoint,
        # but do not appear in the current graph
        in_ckpt_not_in_graph = [ \
                name \
                for name in var_shapes.keys() \
                if name not in all_vars.keys()]
        log.info('Vars in ckpt, not in graph:\n' + str(in_ckpt_not_in_graph))

        # Ensure the vars to restored have the correct shape.
        var_list = {}

        for name, var in restore_vars.items():
            var_shape = var.get_shape().as_list()
            if var_shape == var_shapes[name]:
                var_list[name] = var
            else:
                log.info('Shape mismatch for %s' % name \
                      + str(var_shape) \
                      + str(var_shapes[name]))
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
                var_list=self.var_list,
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
            log.info(inst.args[0] \
                    + "\n Is your dbname too long? "\
                    + "Mongo requires that dbnames be no longer "\
                    + "than 64 characters.")
            log.info("Will not load from cache database.")
            count_recent = 0
            #raise er.OperationFailure(inst.args[0] + "\n Is your dbname too long? Mongo requires that dbnames be no longer than 64 characters.")

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

            # If ndarray found, get the mean of it
            for k, v in train_res.items():
                if k not in ['optimizer', '__grads__'] \
                        and isinstance(v, np.ndarray) \
                        and len(v) > 1 \
                        and k not in self.save_to_gfs:
                    train_res[k] = np.mean(v)

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
            if self.load_data is not None:
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
                recent_query_result = recent_gridfs_files.find(
                        {'saved_filters': True}, sort=[('uploadDate', 1)])
                num_cached_filters = recent_query_result.count()
                cache_max_num = self.cache_max_num
                if num_cached_filters > cache_max_num:
                    log.info('Cleaning up cached filters')
                    fsbucket = gridfs.GridFSBucket(
                            recent_gridfs_files._Collection__database,
                            bucket_name=recent_gridfs_files.name.split('.')[0])

                    for del_indx in range(0, num_cached_filters - cache_max_num):
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
            self.collfs.put(pickle.dumps(save_to_gfs),
                            filename=save_to_gfs_path, item_for=outrec)

        sys.stdout.flush()  # flush the stdout buffer
        self.outrecs.append(outrec)


class CoordinatedThread(threading.Thread):
    """A thread class coordinated by tf.train.Coordinator."""

    def __init__(
            self, coord=None, group=None,
            target=None, name=None, args=(), kwargs={}):
        super(CoordinatedThread, self).__init__(
            group=None, target=target, name=name, args=args, kwargs=kwargs)
        self._coord = coord
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        """Run the thread's main activity.
        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.
        """
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as error:
            self._coord.request_stop(error)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs
