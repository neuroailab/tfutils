"""
This scripts contains some functions that are usually short, easy to understand.
"""
import sys
import collections
import logging
import json
import inspect
import pkg_resources
import os
import re
import copy
import pdb

import numpy as np
from bson.objectid import ObjectId
import git

import tensorflow as tf
from tensorflow.python.client import device_lib
from tfutils.multi_gpu.easy_variable_mgr import REAL_NAME_SCOPE
from tfutils.optimizer import NON_SAVE_SUFFIX


def isstring(x):
    try:
        x + ''
    except:
        return False
    else:
        return True


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


def strip_prefix_from_name(prefix, name):
    prefix = prefix + '/' if not prefix.endswith('/') else prefix
    if name.startswith(prefix):
        name = name[len(prefix):]
        name = strip_prefix_from_name(prefix, name)
    return name


def strip_prefix(prefix, all_vars):
    var_list = {}

    for var in all_vars:
        if isinstance(all_vars, dict):
            new_name = strip_prefix_from_name(prefix, var)
            var_list[new_name] = all_vars[var]
        else:
            new_name = strip_prefix_from_name(prefix, var.op.name)
            var_list[new_name] = var
    return var_list


def get_var_list_wo_prefix(param, var_manager):
    """
    Get all savable variables, strip prefixes
    """
    all_vars = var_manager.savable_variables()
    # Remove all minibatch related unuseful parameters
    all_vars = filter(lambda x: NON_SAVE_SUFFIX not in x.name, all_vars)

    # Strip prefixes added by tfutils
    var_list = strip_prefix(param['model_params']['prefix'], all_vars)
    var_list = strip_prefix(REAL_NAME_SCOPE, var_list)
    return var_list


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
        if tower_outputs[0].shape.ndims == 0:
            for i, output in enumerate(tower_outputs):
                tower_outputs[i] = tf.expand_dims(output, axis=0)
        return tf.concat(tower_outputs, axis=0)

    # Tensorflow variables are not processed.
    elif isinstance(tower_outputs[0], tf.Variable):
        return tower_outputs

    # Dict values are aggregated by key.
    elif isinstance(tower_outputs[0], collections.Mapping):
        return {key: aggregate_outputs([out[key] for out in tower_outputs])
                for key in tower_outputs[0]}

    # List elements are aggregated by index.
    elif isinstance(tower_outputs[0], list):
        return [aggregate_outputs(out) for out in zip(*tower_outputs)]

    # Simply return all other types
    else:
        return tower_outputs


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


# Useful function to average items within one dictionary
def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res
