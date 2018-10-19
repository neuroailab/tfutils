from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple

import six
import tensorflow as tf

from tensorflow.python.ops import data_flow_ops
import pdb


def _all_reduce_using_copy(tensors_across_devices, use_mean):
  """Does an all-reduce of a list of tensors by copying to the current device.

  The tensors are copied to the current device and then reduced.

  Args:
    tensors_across_devices: A list of tensors, each on a different device.
    use_mean: Whether to take the mean of the tensors instead of a sum:
  Returns:
    A reduced tensor on the current device.
  """
  reduced_tensor = tf.add_n(tensors_across_devices)
  if use_mean:
    reduced_tensor *= 1 / len(tensors_across_devices)
  return reduced_tensor


@six.add_metaclass(abc.ABCMeta)
class BatchAllReduceAlgorithm(object):
  """Represents an algorithm for performing a batch all-reduce operation."""

  def batch_all_reduce(self, all_device_tensors):
    """Performs a batch all-reduce.

    The reduction done is a sum.

    `all_device_tensors` is a list of list of tensors that will be batch
    all-reduced. All tensors within a single inner list must be on the same
    device. The nth element in each list, for any n, will be reduced together.
    The return value is in the same form as `all_device_tensors`, except that
    each tensor is reduced.

    For example, if `all_device_tensors` is:
    [[ A,  B  ],     # A and B are on GPU 0
     [ C,  D  ]]     # C and D are on GPU 1

    Then the return value will be:
    [[ A+C,  B+D ],  # These two tensors are on GPU 0
     [ A+C,  B+D ]]  # These two tensors are on GPU 1

    Arguments:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
        is a tensor where `i` is the device index and `j` is the tensor index.

    Returns:
      reduced_all_device_tensors: A list in the same form as
        `all_device_tensors`, except each tensor has been reduced.
      warmup_ops: A list of ops needed to be run once before the all-reduce can
        occur.
    """
    warmup_ops = []
    all_device_tensors = self._do_batch_all_reduce(all_device_tensors)
    return all_device_tensors, warmup_ops

  @abc.abstractmethod
  def _do_batch_all_reduce(self, all_device_tensors):
    """Performs a batch all-reduce.

    Unlike `self.batch_all_reduce`, this does not do any preprocessing of the
    tensors.

    Args:
      all_device_tensors: A list of list of tensors. `all_device_tensors[i][j]`
        is a tensor where `i` is the device index and `j` is the tensor index.
    Returns:
      reduced_all_device_tensors: A list in the same form as
        `all_device_tensors`, except each tensor has been reduced.
    """
    pass


class CopyToDeviceAlgorithm(BatchAllReduceAlgorithm):
  """An algorithm that copies tensors to be reduced to a specific device."""

  def __init__(self, devices_to_reduce_on, use_mean=True):
    self._devices = devices_to_reduce_on
    self._use_mean = use_mean

  def _do_batch_all_reduce(self, all_device_tensors):
    reduced_tensors = []
    for i, tensors_across_devices in enumerate(zip(*all_device_tensors)):
      with tf.device(self._devices[i % len(self._devices)]):
        reduced_tensor = _all_reduce_using_copy(tensors_across_devices,
                                                self._use_mean)
        reduced_tensors.append(reduced_tensor)
    # The tensors will be brought back to each device once they are used.
    return [reduced_tensors] * len(all_device_tensors)


def algorithm_from_params(devices_to_reduce_on):
    """Returns a BatchAllReduceAlgorithm from a Params tuple."""
    return CopyToDeviceAlgorithm(devices_to_reduce_on)
