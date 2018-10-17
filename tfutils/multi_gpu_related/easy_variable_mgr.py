"""Defines VariableMgr and subclasses used to manage variables.

"""

from __future__ import print_function

import re
import tensorflow as tf
import pdb

import easy_batch_allreduce as batch_allreduce

COPY_NAME_SCOPE = 'var_copy_'
REAL_NAME_SCOPE = 'var_copy_0'


class VariableMgr(object):
  """Abstract superclass for class used by BenchmarkCNN to control variables.

    Functions on this class are used to control how variables are created and
    managed, and how gradients are computed and applied.
  """

  def __init__(self, prefix, devices):
    self.devices = devices
    self.staging_delta_ops = []
    self.prefix = prefix

    # A variable for automatic loss scaling.
    self.grad_has_inf_nan = None

  def each_tower_has_variables(self):
    """Returns True if each GPU tower of the model has separate variables."""
    assert False, 'Must be implemented in subclass'

  def supports_staged_vars(self):
    """Whether staged variable management is supported."""
    return False

  def create_outer_variable_scope(self, device_num):
    """Create the tf.variable_scope around all model graph operations."""
    del device_num  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def preprocess_device_grads(self, device_grads):
    """Preprocess the device gradients prior to applying them.

    Args:
      device_grads: List of lists of (gradient, variable) tuples.
        device_grads[t][g] = (gradient, variable), where t is the index of the
        tower and g is the index of the gradient-variable pair.

    Returns: a tuple of (apply_gradients_devices, gradient_state).
      gradient_state is an opaque structure that should be passed to
      get_gradients_to_apply() and append_apply_gradients_ops() (in that order).
      apply_gradients_devices is a list of devices where the gradients will be
      applied with get_gradients_to_apply() and append_apply_gradients_ops().
    """
    del device_grads  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def get_gradients_to_apply(self, device_num, gradient_state):
    """Returns the [(gradient, variable)] list to apply for device_num.

    Args:
      device_num: indexes into apply_gradients_devices, which was returned by an
        earlier call to preprocess_device_grads.
      gradient_state: from previous call to apply_gradients_devices.
    """
    del device_num, gradient_state  # unused by this implementation
    assert False, 'Must be implemented in subclass'

  def get_post_init_ops(self):
    """Returns ops that should run post-initialization."""
    return []

  def get_devices(self):
    """Returns devices to use for computation; includes replica selection."""
    assert False, 'Must be implemented in subclass'

  def savable_variables(self):
    """Returns a list/dict of savable variables to pass to tf.train.Saver."""
    return tf.global_variables()

  def trainable_variables_on_device(self,
                                    abs_device_num,
                                    ):
    """Return the set of trainable variables on device.

    Args:
      abs_device_num: global graph device index.

    Returns:
      The set of trainable variables on the specified device.
    """
    if self.each_tower_has_variables():
      params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('%s/%s%s/' \
                  % (self.prefix, COPY_NAME_SCOPE, abs_device_num))
      ]
    else:
      params = tf.trainable_variables()
    return params


class VariableMgrLocalReplicated(VariableMgr):
  """VariableMgr that implements the --replicated mode for local jobs.

     Each GPU has its own copy of the variables. To apply gradients,
     either a local all-reduce algorithm is applied or a regular
     cross-device aggregation is used to replicate the combined
     gradients to all towers.
  """

  def __init__(
      self, 
      *args,
      **kwargs):
    super(VariableMgrLocalReplicated, self).__init__(*args, **kwargs)
    self._warmup_ops = []
    self._gradient_put_ops = None

  def each_tower_has_variables(self):
    return True

  def create_outer_variable_scope(self, device_num):
    return tf.variable_scope(
        '%s%s' % (COPY_NAME_SCOPE, device_num), 
        use_resource=False)

  def preprocess_device_grads(self, device_grads):

    grads_to_reduce = [[g for g, _ in grad_vars] for grad_vars in device_grads]
    algorithm = batch_allreduce.algorithm_from_params(self.devices)
    reduced_grads, self._warmup_ops \
        = algorithm.batch_all_reduce(grads_to_reduce)
    reduced_device_grads = [[
        (g, v) for g, (_, v) in zip(grads, grad_vars)
        ] for grads, grad_vars in zip(reduced_grads, device_grads)]
    return self.devices, reduced_device_grads

  def get_gradients_to_apply(self, device_num, gradient_state):
    device_grads = gradient_state
    return device_grads[device_num]

  def is_real_tensor(self, tensor):
    split_name = tensor.name.split('/')
    if not tensor.name.startswith('%s/%s' % (self.prefix, COPY_NAME_SCOPE)) \
        or split_name[1] == REAL_NAME_SCOPE:
      return True
    return False

  def get_post_init_ops(self):
    # Copy initialized values for variables on GPU 0 to other GPUs.
    global_vars = tf.global_variables()
    var_by_name = dict([(v.name, v) for v in global_vars])
    post_init_ops = []
    for v in global_vars:
      if self.is_real_tensor(v):
        continue
      split_name = v.name.split('/')
      split_name[1] = REAL_NAME_SCOPE
      copy_from = var_by_name['/'.join(split_name)]
      post_init_ops.append(v.assign(copy_from.read_value()))

    post_init_ops += self._warmup_ops
    return post_init_ops

  def savable_variables(self):
    """Return the set of variables used for saving/loading the model."""
    params = []
    for v in tf.global_variables():
      if self.is_real_tensor(v):
        params.append(v)
    return params

  def get_devices(self):
    return self.devices
