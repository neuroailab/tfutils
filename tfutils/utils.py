import sys
import collections
import logging
import json
import datetime
import inspect
import threading
import pkg_resources
import os
import re
import copy

import numpy as np
from bson.objectid import ObjectId
import git

import tensorflow as tf
from tensorflow.python import DType
from tensorflow.python.client import device_lib


def isstring(x):
    try:
        x + ''
    except:
        return False
    else:
        return True


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


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def format_devices(devices):
    """Return list of proper device (gpu) strings.

    Args:
        devices (list): A list of device strings.
            If ``devices`` is not a list, it is converted to one. The following
            rules are applied to each element in the list:

                type: (int) -> '/gpu:{}'.format(int)
                type: (str) -> /gpu:{}'.format(d) where d is first occurence
                of a digit

    Returns:
        list: A sorted list of unique and properly formatted device strings.

    Raises:
        TypeError: Invalid device specification.

    """
    def format_device(device):
        gpu = '/gpu:{}'
        if isinstance(device, int):
            gpu = gpu.format(device)
        else:
            m = re.search(r'/gpu:\d+$', device)
            n = re.search(r'\d+', device)
            if m is not None:
                gpu = m.group()
            elif n is not None:
                gpu = gpu.format(n.group())
            else:
                raise TypeError(
                    'Invalid device specification: {}'.format(device))
        return gpu

    devices = [devices] if not isinstance(devices, list) else devices
    return sorted(list(set(map(format_device, devices))))


def strip_prefix(prefix, all_vars):

    # def _strip_prefix_from_name(prefix, name):
    #     prefix = prefix + '/' if not prefix.endswith('/') else prefix
    #     if name.startswith(prefix):
    #         name = name[len(prefix):]
    #         name = _strip_prefix_from_name(prefix, name)
    #     return name

    var_list = {}

    for var in all_vars:
        new_name = strip_prefix_from_name(prefix, var.op.name)
        var_list[new_name] = var
    return var_list


def strip_prefix_from_name(prefix, name):
    prefix = prefix + '/' if not prefix.endswith('/') else prefix
    if name.startswith(prefix):
        name = name[len(prefix):]
        name = strip_prefix_from_name(prefix, name)
    return name


def aggregate_outputs(tower_outputs):
    """Return aggregated model replicate outputs.

    The elements of `tower_outputs` should have identical structure and
    correspond to the outputs of individual model replicas on separate
    devices (GPUs). Model replicate outputs are recursively searched until
    a tensor ``t`` satisfying:

        isinstance(t, tf.Tensor) -> True

    is found. Tensor ``t`` is then concatenated with all of its corresponding
    replicates along the batch dimension (axis=0).

    If ``tower_outputs`` is a list of length one, then the element it contains
    is returned.

    Args:
        tower_outputs (list): The outputs of individual model replicas.

    Returns:
        The aggregated output with a structure identical to the replicate outputs.

    Raises:
        TypeError: Aggregation not supported for given type.

    Examples:
        >>> print(tower_outputs)
        [{'tensor': <tf.Tensor 'softmax_linear/fc/output:0' shape=(50, 10) dtype=float32>},
        {'tensor': <tf.Tensor 'softmax_linear_1/fc/output:0' shape=(50, 10) dtype=float32>}]
        >>>
        >>> print(aggegrate_ouputs(tower_outputs))
        {'tensor': <tf.Tensor 'concat:0' shape=(100, 10) dtype=float32>}

    """
    if len(tower_outputs) == 1:
        return tower_outputs.pop()

    # Tensorflow tensors are concatenated along axis 0.
    elif isinstance(tower_outputs[0], tf.Tensor):
        return tf.concat(tower_outputs, axis=0)

    # Dict values are aggregated by key.
    elif isinstance(tower_outputs[0], collections.Mapping):
        return {key: aggregate_outputs([out[key] for out in tower_outputs])
                for key in tower_outputs[0]}

    # List elements are aggregated by index.
    elif isinstance(tower_outputs[0], collections.Iterable):
        return [aggregate_outputs(out) for out in zip(*tower_outputs)]

    # All other types are not supported.
    raise TypeError('Aggregation not supported for type: {}'.
                    format(type(tower_outputs[0])))


def get_loss(inputs,
             outputs,
             targets,
             loss_per_case_func,
             loss_per_case_func_params={'_outputs': 'logits', '_targets_$all': 'labels'},
             agg_func=None,
             loss_func_kwargs=None,
             agg_func_kwargs=None,
             **loss_params):
    if loss_func_kwargs is None:
        loss_func_kwargs = {}
    else:
        loss_func_kwargs = copy.deepcopy(loss_func_kwargs)
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
                loss_func_kwargs[loss_per_case_func_params[
                    key_value]] = inputs[tmp_key]

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



def identity_func(x):
    if not hasattr(x, 'keys'):
        x = {'result': x}
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
    """An immuatable dictionary.

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


def predict(step, results):
    if not hasattr(results['output'], '__iter__'):
        outputs = [results['outputs']]
    else:
        outputs = results['outputs']

    preds = [tf.argmax(output, 1) for output in outputs]

    return preds
