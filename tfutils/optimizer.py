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
import numpy as np
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
        Alternatively, it will be a list, where each element is the optimizer_class.
        clip (bool, optional): Default is True, clipping by `[-1, 1]`.

    """
    def __init__(
            self, optimizer_class, learning_rate, global_step=None, 
            clip=True, clipping_method='value', clipping_value=1.0, 
            print_global_norm=False, trainable_scope=None, 
            optimizer_args=None, optimizer_kwargs=None):

        if not isinstance(optimizer_class, list):
            self._optimizer_class = [optimizer_class]
        else:
            self._optimizer_class = optimizer_class

        if optimizer_args is None:
            self._optimizer_args = [{}]*len(self._optimizer_class)
        elif not isinstance(optimizer_args, list):
            assert(len(self._optimizer_class) == 1)
            self._optimizer_args = [optimizer_args]
        else:
            self._optimizer_args = optimizer_args
        assert(len(self._optimizer_class) == len(self._optimizer_args))

        if optimizer_kwargs is None:
            self._optimizer_kwargs = [{}]*len(self._optimizer_class)
        elif not isinstance(optimizer_kwargs, list):
            assert(len(self._optimizer_class) == 1)
            self._optimizer_kwargs = [optimizer_kwargs]
        else:
            self._optimizer_kwargs = optimizer_kwargs
        assert(len(self._optimizer_class) == len(self._optimizer_kwargs))

        if not isinstance(learning_rate, list):
            assert(len(self._optimizer_class) == 1)
            self._learning_rate = [learning_rate]
        else:
            self._learning_rate = learning_rate
        assert(len(self._optimizer_class) == len(self._learning_rate))

        self._optimizers = []
        for opt_idx, opt_cls in enumerate(self._optimizer_class):
            curr_opt_args = self._optimizer_args[opt_idx]
            curr_opt_kwargs = copy.deepcopy(self._optimizer_kwargs[opt_idx])
            curr_opt_kwargs['learning_rate'] = self._learning_rate[opt_idx]
            if self._optimizer_kwargs[opt_idx].get('include_global_step', False):
            	curr_opt_kwargs.pop('include_global_step', None)
            	curr_opt_kwargs['global_step'] = global_step
            curr_opt_func = opt_cls(*curr_opt_args, **curr_opt_kwargs)
            self._optimizers.insert(opt_idx, curr_opt_func)

            # The optimizer needs to have these required methods
            required_methods = ['compute_gradients', 'apply_gradients']
            for required_method in required_methods:
                assert required_method in dir(curr_opt_func), \
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
        if not isinstance(loss, list):
            loss = [loss]
        assert(len(loss) == len(self._optimizer_class))

        if self.trainable_scope is not None:
            new_var_list = [v for v in var_list if any([nm in v.name for nm in self.trainable_scope])]
            if len(new_var_list):
                var_list = new_var_list
                log.info("Only training variables in scope: %s" % self.trainable_scope)
                log.info("variables to be trained: %s" % var_list)

        if var_list is not None:
            num_trainable_params = sum([np.prod(v.shape.as_list()) for v in var_list])
            log.info("Number of Trainable Parameters: %d" % num_trainable_params)

        gvs_list = []
        for opt_idx, curr_opt_func in enumerate(self._optimizers):

            gvs = curr_opt_func.compute_gradients(loss[opt_idx], var_list=var_list,
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
                    gradients, global_norm = tf.clip_by_global_norm(gradients, self.clipping_value,
                            use_norm=norm)
                    gvs = zip(gradients, variables)
                else:
                    raise ValueError("optimizer.clip = True but you didn't specify a valid method in ['value', 'norm']")

            gvs_list.insert(opt_idx, gvs)

        return gvs_list

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
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
        assert(len(grads_and_vars) == len(self._optimizer_class)) # make sure it is consistent
        optimize_op_list = []
        for opt_idx, curr_opt_func in enumerate(self._optimizers):
            optimize = curr_opt_func.apply_gradients(grads_and_vars[opt_idx],
                                                   global_step=global_step,
                                                   name=name)
            optimize_op_list.insert(opt_idx, optimize)

        return tf.group(*optimize_op_list)

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
        self._multi_mode = (type(self._optimizer).__name__ == 'ClipOptimizer') # list of lists, grads and vars per optimizer in this case

    def filter_none_vars(self, gvs, opt_idx=None):
        gvs_wo_none = []
        for grad, var in gvs:
            if grad is not None:
                gvs_wo_none.append([grad, var])
        curr_var_list = [each_var for _, each_var in gvs_wo_none]
        if opt_idx is None:
            self.var_list = curr_var_list
        else:
            if opt_idx == 0:
                self.var_list = [curr_var_list]
            elif opt_idx > 0:
                self.var_list.insert(opt_idx, curr_var_list)
            else:
                raise ValueError
        return gvs_wo_none

    def compute_gradients(self, loss, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(
                loss,
                *args, **kwargs)
        # Get the variables to update from results of compute_gradients
        # filter out the variables with None
        
        if self._multi_mode: # list of lists, grads and vars per optimizer
            gvs_wo_none = []
            for opt_idx, curr_gv in enumerate(gvs):
                curr_gv_wo_none = self.filter_none_vars(curr_gv, opt_idx=opt_idx)
                gvs_wo_none.insert(opt_idx, curr_gv_wo_none)
        else:
            gvs_wo_none = self.filter_none_vars(gvs)
        return gvs_wo_none

    def _consistency_check(self, curr_mb_gv, opt_idx=None):
        # Make sure that the var_list is the same variable list with
        # that in minibatch_grads
        if opt_idx is None:
            curr_var_list = self.var_list
        else:
            curr_var_list = self.var_list[opt_idx]

        assert len(curr_mb_gv) == len(curr_var_list), \
                "Variable list length not matched!"
        assert all((\
                var_g.name == var_l.name \
                for (_, var_g), var_l in zip(curr_mb_gv, curr_var_list))),\
                "Variable list should have the same variables!" 

    def _zero_gvs(self, opt_idx=None):
        if opt_idx is None:
            curr_var_list = self.var_list
        else:
            curr_var_list = self.var_list[opt_idx]

        zero_gvs = [(
                    tf.Variable(
                        tf.zeros_like(var.initialized_value()),
                        dtype=tf.float32,
                        trainable=False,
                        name=NON_SAVE_SUFFIX,
                        ),
                    var) for var in curr_var_list]
        return zero_gvs

    def _mini_ops(self, grads_and_vars, minibatch_grads, num_minibatches=1):
        mini_ops = []
        for (grad_v, _), (mini_grad, _) \
                in zip(grads_and_vars, minibatch_grads):
            mini_ops.append(
                    tf.assign_add(grad_v, mini_grad / num_minibatches))
        return mini_ops

    def accumulate_gradients(self, minibatch_grads, num_minibatches=1):
        """Accumulate gradients for `num_minibatches` minibatches."""
        if num_minibatches == 1:
            # No need for accumulating
            return tf.no_op(), minibatch_grads

        if self._multi_mode: # list of lists, grads and vars per optimizer
            assert(len(minibatch_grads) == len(self._optimizer._optimizer_class))
            for opt_idx, curr_mb_gv in enumerate(minibatch_grads):
                self._consistency_check(curr_mb_gv, opt_idx=opt_idx)

            if self.grads_and_vars is None:
                zero_gvs_list = []
                for opt_idx, _ in enumerate(minibatch_grads):
                    zero_gvs_list.insert(opt_idx, self._zero_gvs(opt_idx=opt_idx))
                self.grads_and_vars = zero_gvs_list

            mini_ops_list = []
            for opt_idx, curr_mb_gv in enumerate(minibatch_grads):
                curr_mini_ops = self._mini_ops(grads_and_vars=self.grads_and_vars[opt_idx], minibatch_grads=curr_mb_gv, num_minibatches=num_minibatches)
                mini_ops_list.extend(curr_mini_ops)
            mnb_accu_grad = tf.group(*mini_ops_list)

        else:
            self._consistency_check(minibatch_grads)
            if self.grads_and_vars is None:
                self.grads_and_vars = self._zero_gvs()

            mini_ops = self._mini_ops(grads_and_vars=self.grads_and_vars, minibatch_grads=minibatch_grads, num_minibatches=num_minibatches)
            mnb_accu_grad = tf.group(*mini_ops)

        return mnb_accu_grad, self.grads_and_vars

    def _reset_grads(self, grads_and_vars):
        reset_ops = []
        for grad_v, _ in grads_and_vars:
            reset_ops.append(tf.assign(grad_v, tf.zeros(grad_v.shape)))
        return reset_ops

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
            assert(len(self.grads_and_vars) == len(self._optimizer._optimizer_class))
            with tf.control_dependencies([optimize]):
                if self._multi_mode:
                    reset_ops = []
                    for opt_idx, _ in enumerate(self.grads_and_vars):
                        curr_reset_ops = self._reset_grads(grads_and_vars=self.grads_and_vars[opt_idx])
                        reset_ops.extend(curr_reset_ops)
                else:
                    reset_ops = self._reset_grads(grads_and_vars=self.grads_and_vars)
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
