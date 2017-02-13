import collections
import logging
import json
import datetime
import inspect
import pkg_resources
import os
import re
import copy

import numpy as np
from bson.objectid import ObjectId
import git

# from tfutils.error import RepoIsDirtyError

logging.basicConfig()
log = logging.getLogger('tfutils')


def isstring(x):
    try:
        x + ''
    except:
        return False
    else:
        return True
        

def version_info(module):
    """Gets version of a standard python module
    """
    if hasattr(module, '__version__'):
        version = module.__version__
    elif hasattr(module, 'VERSION'):
        version = module.VERSION
    else:
        pkgname = module.__name__.split('.')[0]
        try:
            info = pkg_resources.get_distribution(pkgname)
        except pkg_resources.DistributionNotFound:
            version = None
            log.warning('version information not found for %s' % module.__name__)
        else:
            version = info.version

    return {'version': version}


def version_check_and_info(module):
    """returns either git information or standard module version if not a git repo

    Args: - module (module): python module object to get info for.
    Returns: dictionary of info
    """
    srcpath = inspect.getsourcefile(module)
    try:
        repo = git.Repo(srcpath, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        log.info('module %s not in a git repo, checking package version' % module.__name__)
        info = version_info(module)
    else:
        info = git_info(repo)
    info['source_path'] = srcpath
    return info


def git_info(repo):
    """information about a git repo
    """
    if repo.is_dirty():
        log.warning('repo %s is dirty' % repo.git_dir)
        clean = False
    else:
        clean = True
    branchname = repo.active_branch.name
    commit = repo.active_branch.commit.hexsha
    origin = repo.remote('origin')
    urls = map(str, list(origin.urls))
    remote_ref = [_r for _r in origin.refs if _r.name == 'origin/' + branchname]
    if not len(remote_ref) > 0:
        log.warning('Active branch %s not in origin ref' % branchname)
        active_branch_in_origin = False
        commit_in_log = False
    else:
        active_branch_in_origin = True
        remote_ref = remote_ref[0]
        gitlog = remote_ref.log()
        shas = [_r.oldhexsha for _r in gitlog] + [_r.newhexsha for _r in gitlog]
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


def make_mongo_safe(_d):
    """Makes a json-izable actually safe for insertion into Mongo.
    """
    klist = _d.keys()[:]
    for _k in klist:
        if hasattr(_d[_k], 'keys'):
            make_mongo_safe(_d[_k])
        if not isinstance(_k, str):
            _d[str(_k)] = _d.pop(_k)
        _k = str(_k)
        if '.' in _k:
            _d[_k.replace('.', '___')] = _d.pop(_k)


def sonify(arg, memo=None):
    """when possible, returns version of argument that can be
       serialized trivally to json format
    """
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]

    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, datetime.datetime):
        rval = arg
    elif isinstance(arg, np.floating):
        rval = float(arg)
    elif isinstance(arg, np.integer):
        rval = int(arg)
    elif isinstance(arg, (list, tuple)):
        rval = type(arg)([sonify(ai, memo) for ai in arg])
    elif isinstance(arg, collections.OrderedDict):
        rval = collections.OrderedDict([(sonify(k, memo), sonify(v, memo))
                                        for k, v in arg.items()])
    elif isinstance(arg, dict):
        rval = dict([(sonify(k, memo), sonify(v, memo))
                     for k, v in arg.items()])
    elif isinstance(arg, (basestring, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = sonify(arg.sum())
        else:
            rval = map(sonify, arg)  # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    elif callable(arg):
        mod = inspect.getmodule(arg)
        modname = mod.__name__
        objname = arg.__name__
        rval = version_check_and_info(mod)
        rval.update({'objname': objname,
                     'modname': modname})
        rval = sonify(rval)
    else:
        raise TypeError('sonify', arg)

    memo[id(rval)] = rval
    return rval


def jsonize(x):
    """returns version of x that can be serialized trivally to json format
    """
    try:
        json.dumps(x)
    except TypeError:
        return sonify(x)
    else:
        return x


def get_loss(inputs,
             outputs,
             targets,
             loss_per_case_func,
             loss_per_case_func_params={'_outputs': 'logits', '_targets_$all': 'labels'},
             agg_func=None,
             loss_func_kwargs=None,
             agg_func_kwargs=None):
    if loss_func_kwargs is None:
        loss_func_kwargs = {}
    if not isinstance(targets, (list, tuple, np.ndarray)):
        targets = [targets]
    targets = list(targets)
    if len(targets) == 1:
        labels = inputs[targets[0]]
    else:
        labels = [inputs[t] for t in targets]

    flag_with_out = True
    flag_with_tar = True
    for key_value in loss_per_case_func_params.keys():
        if key_value == '_outputs':
            flag_with_out = False
            loss_func_kwargs[loss_per_case_func_params[key_value]] = outputs
        elif key_value == '_targets_$all':
            flag_with_tar = False
            loss_func_kwargs[loss_per_case_func_params[key_value]] = labels
        elif key_value.startswith('_targets_'):
            tmp_key = key_value[len('_targets_'):]
            if tmp_key in targets:
                targets.remove(tmp_key)
                loss_func_kwargs[loss_per_case_func_params[key_value]] = inputs[tmp_key]

    if len(targets) == 0:
        flag_with_tar = False
    elif len(targets) == 1:
        labels = inputs[targets[0]]
    else:
        labels = [inputs[t] for t in targets]

    if not flag_with_tar:
        labels = []
    if not isinstance(labels, (list, tuple, np.ndarray)):
        labels = [labels]
    if flag_with_out:
        labels.insert(0, outputs)
    loss = loss_per_case_func(*labels, **loss_func_kwargs)

    if agg_func is not None:
        if agg_func_kwargs is None:
            agg_func_kwargs = {}
        loss = agg_func(loss, **agg_func_kwargs)
    return loss


def get_loss_dict(*args, **kwargs):
    kwargs = copy.copy(kwargs)
    name = kwargs.pop('name', 'loss')
    return {name: get_loss(*args, **kwargs)}


def verify_pb2_v2_files(cache_prefix, ckpt_record):
    file_data = get_saver_pb2_v2_files(cache_prefix)
    ndf = file_data['num_data_files']
    sndf = ckpt_record['_saver_num_data_files']
    assert ndf == sndf, (ndf, sndf)


def get_saver_pb2_v2_files(prefix):
    dirn, pref = os.path.split(prefix)
    pref = pref + '.'
    files = filter(lambda x: x.startswith(pref) and not x.endswith('.tar'),
                   os.listdir(dirn))
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
    assert fns == range(total0), (fns, total0)
    files = [os.path.join(dirn, f) for f in files]
    file_data = {'files': files, 'num_data_files': total0}
    return file_data


def identity_func(x):
    return x


def append_and_return(x, y, step):
    if x is None:
        x = []
    x.append(y)
    return x


def reduce_mean(x, y, step):
    if x is None:
        x = y
    else:
        f = step / (step + 1.)
        x = f * x + (1 - f) * y
    return x


def reduce_mean_dict(x, y, step):
    if x is None:
        x = {}
    for k in y:
        # ka = k + '_agg'
        ka = k
        if k != 'validation_step':
            x[ka] = reduce_mean(x.get(ka), y[k], step)
        else:
            x[ka] = [min(min(x[ka]), y[k]), max(max(x[ka]), y[k])]
    return x


def mean_dict(y):
    x = {}
    keys = y[0].keys()
    for k in keys:
        # ka = k + '_agg'
        ka = k
        pluck = [_y[k] for _y in y]
        if k != 'validation_step':
            x[ka] = np.mean(pluck)
        else:
            x[ka] = [min(pluck), max(pluck)]
    return x


class frozendict(collections.Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability is desired.

    from https://pypi.python.org/pypi/frozendict

    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash
