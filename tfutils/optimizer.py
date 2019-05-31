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

NON_SAVE_SUFFIX = '__tfutils_minibatch__'


class ClipOptimizer(object):
    """A wrapper for general optimizers.

    This class supports:

    - Clipping the gradients. (controlled by clip parameter)
    - Train part of trainable parameters (controlled by trainable_names)

    Args:
        optimizer_class: Returned value of this function should have `compute_gradients` and `apply_gradients` methods.
        clip (bool, optional): Default is True, clipping by `[-1, 1]`.

    """
    def __init__(
            self, optimizer_class, clip=True, clipping_method='value', clipping_value=1.0, print_global_norm=False,
            trainable_scope=None, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        # The optimizer needs to have these required methods
        required_methods = ['compute_gradients', 'apply_gradients']
        for required_method in required_methods:
            assert required_method in dir(self._optimizer), \
                    "Your optimizer needs to have method %s!" % required_method

        self.clip = clip
        self.clipping_method = clipping_method
        self.clipping_value = clipping_value
        self.print_global_norm = print_global_norm
        self.trainable_scope = trainable_scope

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        """Compute gradients to model variables from loss.

        Args:
            loss (tf.Tensor): Tensorflow loss to optimize.

        Returns:
            (tf.Operation): Compute gradient update to model followed by a
            clipping operation if `self.clip` is True.

        """
        # freeze all variables except those with self.trainable_scope in their names
        if self.trainable_scope is not None:
            new_var_list = [v for v in var_list if self.trainable_scope in v.name]
            if len(new_var_list):
                var_list = new_var_list
                log.info("Only training variables in scope: %s" % self.trainable_scope)            

        gvs = self._optimizer.compute_gradients(loss, var_list=var_list,
                                                *args, **kwargs)

        if self.clip:
            if self.clipping_method == "value":
                # gradient clipping. Some gradients returned are 'None' because
                # no relation between the variable and loss; so we skip those.
                gvs = [(tf.clip_by_value(grad, -self.clipping_value, self.clipping_value), var)
                        for grad, var in gvs if grad is not None]
            elif self.clipping_method == "norm":
                print("USING GLOBAL NORM CLIPPING with clip_value %.2f" % self.clipping_value)
                gradients, variables = zip(*gvs)
                norm = tf.global_norm(gradients)
                if self.print_global_norm:
                    norm = tf.Print(norm, [norm], message="grad_global_norm")
                true_fn = lambda: tf.constant(1.0)
                false_fn = lambda: tf.identity(norm)
                norm = tf.case([(tf.logical_or(tf.is_inf(norm), tf.is_nan(norm)), true_fn)], default=false_fn)                
                # norm = tf.case([(tf.is_nan(norm), true_fn)], default=false_fn)
                gradients, global_norm = tf.clip_by_global_norm(gradients, self.clipping_value,
                        use_norm=norm)
                gvs = zip(gradients, variables)
            else:
                raise ValueError("optimizer.clip = True but you didn't specify a valid method in ['value', 'norm']")
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
            return tf.no_op(), minibatch_grads

        # Make sure that the var_list is the same variable list with
        # that in minibatch_grads
        assert len(minibatch_grads) == len(self.var_list), \
                "Variable list length not matched!"
        assert all((\
                var_g.name == var_l.name \
                for (_, var_g), var_l in zip(minibatch_grads, self.var_list))),\
                "Variable list should have the same variables!"
        if self.grads_and_vars is None:
            self.grads_and_vars = [(
                tf.Variable(
                    tf.zeros_like(var.initialized_value()),
                    dtype=tf.float32,
                    trainable=False,
                    name=NON_SAVE_SUFFIX,
                    ),
                var) for var in self.var_list]

        mini_ops = []
        for (grad_v, _), (mini_grad, _) \
                in zip(self.grads_and_vars, minibatch_grads):
            mini_ops.append(
                    tf.assign_add(grad_v, mini_grad / num_minibatches))
        mnb_accu_grad = tf.group(*mini_ops)

        return mnb_accu_grad, self.grads_and_vars

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
        optimize = self._optimizer.apply_gradients(
                grads_and_vars,
                global_step=global_step)

        # Zero the stored grads if needed
        if self.grads_and_vars is not None:
            with tf.control_dependencies([optimize]):
                reset_ops = []
                for grad_v, _ in self.grads_and_vars:
                    reset_ops.append(tf.assign(grad_v, tf.zeros(grad_v.shape)))
                reset_act = tf.group(*reset_ops)
            return reset_act

        return optimize

    def accu_and_apply_grads(
            self, minibatch_grads,
            num_minibatch, global_step):
        # Aggregate and accumulate gradients.
        mnb_accu_grad, grads_and_vars = self.accumulate_gradients(
                minibatch_grads,
                num_minibatch)

        # Apply accumulated gradients.
        optimizer = self.apply_gradients(grads_and_vars, global_step)

        return mnb_accu_grad, optimizer
