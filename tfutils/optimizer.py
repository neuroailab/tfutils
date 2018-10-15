"""Default Optimizer to be used with tfutils.

The ClipOptimizer class adds support for gradient clipping, self-defined 
trainable parameters. This optimizer is just a tool provided by TFUtils, 
but not what tfutils must use.

The MinibatchOptimizer adds support for gradient aggregation gradient 
accumulation useful for performing minibatching (accumulating and aggregating
gradients for multiple batches before applying a gradient update). 
This optimizer is what tfutils must use.

"""
import os
import copy
import tensorflow as tf
import logging
import pdb

if 'TFUTILS_LOGFILE' in os.environ:
    logging.basicConfig(filename=os.environ['TFUTILS_LOGFILE'])
    print ("USING LOGFILE: %s" % os.environ['TFUTILS_LOGFILE'])
else:
    logging.basicConfig()

log = logging.getLogger('tfutils')
log.setLevel('DEBUG')


class ClipOptimizer(object):
    """A wrapper for general optimizers. 

    This class supports:

    - Clipping the gradients. (controlled by clip parameter)
    - Train part of trainable parameters (controlled by trainable_names)

    Args:
        optimizer_class: Returned value of this function should have `compute_gradients` and `apply_gradients` methods.
        clip (bool, optional): Default is True, clipping by `[-1, 1]`.
        trainable_names (list of strings, or string, optional): Default is None. Scope names for variables to avoid training.

    """
    def __init__(
            self, optimizer_class, clip=True, 
            trainable_names=None, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        # The optimizer needs to have these required methods
        required_methods = ['compute_gradients', 'apply_gradients']
        for required_method in required_methods:
            assert required_method in dir(self._optimizer), \
                    "Your optimizer needs to have method %s!" % required_method

        self.clip = clip
        if not isinstance(trainable_names, list) \
                and trainable_names is not None:
            trainable_names = [trainable_names]
        self.trainable_names = trainable_names

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        """Compute gradients to model variables from loss.

        Args:
            loss (tf.Tensor): Tensorflow loss to optimize.

        Returns:
            (tf.Operation): Compute gradient update to model followed by a
            clipping operation if `self.clip` is True.

        """
        train_vars = var_list
        if self.trainable_names is not None:
            if train_vars:
                log.info('Ignoring specified trainable_names!')
            else:
                train_key = tf.GraphKeys.TRAINABLE_VARIABLES
                log.info('All trainable vars:\n' \
                        + str([var.name for var in tf.get_collection(train_key)]))
                train_vars = []
                for scope_name in self.trainable_names:
                    new_vars = tf.get_collection(
                            train_key, 
                            scope=scope_name)
                    if len(new_vars) == 0:
                        raise ValueError(
                                'The scope name, {}, '.format(scope_name)\
                                + 'you specified does not contain '\
                                + 'any trainable variables.')
                    train_vars.extend(new_vars)
                log.info('Variables to be trained:\n' \
                        + str([var.name for var in train_vars]))

        gvs = self._optimizer.compute_gradients(loss,
                                                var_list=train_vars,
                                                *args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gvs if grad is not None]
        return gvs

    def apply_gradients(self, grads_and_vars, global_step=None):
        """Apply gradients to model variables specified in `grads_and_vars`.

        `apply_gradients` returns an op that calls
        `tf.train.Optimizer.apply_gradients`

        Args:
            grads_and_vars (list): Description.
            global_step (None, optional): tensorflow global_step variable.

        Returns:
            (tf.Operation): Applies gradient update to model followed by an
                internal gradient zeroing operation to `self.grads_and_vars`.

        """
        optimize = self._optimizer.apply_gradients(grads_and_vars,
                                                   global_step=global_step)
        return optimize


class MinibatchOptimizer(object):
    """A wrapper used by tfutils for general optimizers. 

    This class supports:

    - Minibatch, only apply gradients after several steps. By default, apply gradients after each step
    """

    def __init__(self, optimizer, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer(*optimizer_args, **optimizer_kwargs)
        # The optimizer needs to have these required methods
        required_methods = ['compute_gradients', 'apply_gradients']
        for required_method in required_methods:
            assert required_method in dir(self._optimizer), \
                    "Your optimizer needs to have method %s!" % required_method

        self.grads_and_vars = None
        self.mini_flag = tf.Variable(tf.zeros(1), trainable=False)

    def compute_gradients(self, loss, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(
                loss,
                *args, **kwargs)
        # Get the variables to update from results of compute_gradients
        # filter out the variables with None
        gvs_wo_none = []
        for grad, var in gvs:
            if grad is not None:
                gvs_wo_none.append([grad, var])
        gvs = gvs_wo_none
        self.var_list = [each_var for _, each_var in gvs]
        return gvs

    def accumulate_gradients(self, minibatch_grads, num_minibatches=1):
        """Accumulate gradients for `num_minibatches` minibatches."""
        if num_minibatches == 1:
            # No need for accumulating
            return self.mini_flag, minibatch_grads

        if self.grads_and_vars is None:
            self.grads_and_vars = [(
                tf.Variable(tf.zeros_like(var.initialized_value()),
                            dtype=tf.float32,
                            trainable=False),
                var) for var in self.var_list]

        # Add 1/num_minibatches * minibatch_grads to current gradients.
        def _add_op(gv_tmp, mgv_tmp):
            return tf.add(gv_tmp, tf.divide(mgv_tmp, num_minibatches))
        def _set_op(gv_tmp, mgv_tmp):
            return tf.assign(gv_tmp, tf.divide(mgv_tmp, num_minibatches))

        # mini_flag indicates whether it's the first minibatch, 
        # which requires setting up the initial values
        only_grads = [\
                tf.cond(
                    tf.less(self.mini_flag[0], 0.5), 
                    lambda: _set_op(gv[0], mgv[0]), 
                    lambda: _add_op(gv[0], mgv[0])) \
                if mgv[0] is not None else None \
                for (gv, mgv) in zip(self.grads_and_vars, minibatch_grads)]
        only_grads_without_None = []
        for curr_value in only_grads:
            if curr_value is not None:
                only_grads_without_None.append(curr_value)

        with tf.control_dependencies(only_grads_without_None):
            # Set mini_flag to 1, indicates that the initial values are set
            self.mini_flag = tf.assign(
                    self.mini_flag, tf.constant([1], dtype = tf.float32))

        # Filter out the None values here
        grads = []
        for (gv, only_grad) in zip(self.grads_and_vars, only_grads):
            if only_grad is not None:
                grads.append((only_grad, gv[1]))
        return self.mini_flag, grads

    def apply_gradients(self, grads_and_vars, global_step=None):
        """Apply gradients to model variables specified in `grads_and_vars`.

        `apply_gradients` returns an op that calls
        `tf.train.Optimizer.apply_gradients`.

        Args:
            grads_and_vars (list): Description.
            global_step (None, optional): tensorflow global_step variable.

        Returns:
            (tf.Operation): Applies gradient update to model followed by an
            internal gradient zeroing operation to `self.grads_and_vars`.

        """
        # Set back mini_flag as apply_gradients is only called at the end of of batch
        self.mini_flag = tf.assign(
                self.mini_flag, 
                tf.constant([0], dtype = tf.float32))
        with tf.control_dependencies([self.mini_flag]):
            optimize = self._optimizer.apply_gradients(grads_and_vars,
                                                       global_step=global_step)
        return optimize

    def accu_and_apply_grads(
            self, minibatch_grads, 
            num_minibatch, global_step):
        # Aggregate and accumulate gradients.
        mini_flag, grads = self.accumulate_gradients(
                minibatch_grads, 
                num_minibatch)

        # Apply accumulated gradients.
        optimizer = self.apply_gradients(grads, global_step)

        return mini_flag, optimizer
