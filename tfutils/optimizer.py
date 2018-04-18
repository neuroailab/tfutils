"""Default Optimizer to be used with tfutils.

The ClipOptimizer class adds support for gradient clipping, gradient
aggregation across devices and gradient accumulation useful for
performing minibatching (accumulating and aggregating
gradients for multiple batches before applying a gradient update).

"""
import copy
import tensorflow as tf
import logging

logging.basicConfig()
log = logging.getLogger('tfutils')
log.setLevel('DEBUG')

class ClipOptimizer(object):

    def __init__(self, optimizer_class, clip=True, trainable_names=None, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip
        self.var_list = None
        if not isinstance(trainable_names, list) and trainable_names is not None:
            trainable_names = [trainable_names]
        self.trainable_names = trainable_names
        self.grads_and_vars = None

        self.mini_flag = tf.Variable(tf.zeros(1), trainable=False)

    def compute_gradients(self, loss, *args, **kwargs):
        train_vars = None
        if self.trainable_names is not None:
            log.info('All trainable vars:\n'+str([var.name for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
            train_vars = []
            for scope_name in self.trainable_names:
                new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
                if len(new_vars) == 0:
                    raise ValueError('The scope name, {}, you specified does not contain any trainable variables.'.format(scope_name))
                train_vars.extend(new_vars)
            log.info('Variables to be trained:\n'+str([var.name for var in train_vars]))
        if train_vars is not None:
            self.var_list = train_vars

        gvs = self._optimizer.compute_gradients(loss,
                                                var_list=train_vars,
                                                *args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   if grad is not None else (grad, var) for grad, var in gvs]
        for grad, var in gvs:
            assert grad is not None, (grad, var)
        return gvs

    @classmethod
    def aggregate_gradients(cls, grads_and_vars, method='average'):
        if method == 'average':
            return cls.average_gradients(grads_and_vars)
        else:
            raise ValueError('Unsupported aggregation method: {}.'.format(method))

    @classmethod
    def average_gradients(cls, tower_grads):
        """Average a list of (grads, vars) produced by `compute_gradients`."""
        average_grads = []
        for grads_and_vars in zip(*tower_grads):
            # print(grads_and_vars)
            grads = []
            for g, _ in grads_and_vars:
                # print(g.get_shape().as_list(), g)
                grads.append(tf.expand_dims(g, axis=0))
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            # all variables are the same so we just use the first gpu variables
            var = grads_and_vars[0][1]
            grad_and_var = (grad, var)
            average_grads.append(grad_and_var)
        return average_grads

    def accumulate_gradients(self, minibatch_grads, num_minibatches=1):
        """Accumulate gradients for `num_minibatches` minibatches."""
        if self.var_list is None:
            self.var_list = tf.trainable_variables()

        if self.grads_and_vars is None:
            self.grads_and_vars = [(
                tf.Variable(tf.zeros_like(var.initialized_value()),
                            dtype=tf.float32,
                            trainable=False),
                var) for var in self.var_list]

        # Add 1/num_minibatches * minibatch_grads to current gradients.

        if num_minibatches>1:
            def _add_op(gv_tmp, mgv_tmp):
                return tf.add(gv_tmp, tf.divide(mgv_tmp, num_minibatches))
            def _set_op(gv_tmp, mgv_tmp):
                return tf.assign(gv_tmp, tf.divide(mgv_tmp, num_minibatches))
            #grads = [(gv[0].assign_add(tf.divide(mgv[0], num_minibatches)), gv[1])
            #         for (gv, mgv) in zip(self.grads_and_vars, minibatch_grads)]
            #grads = tf.cond(tf.less(self.mini_flag[0], 0.5), fn1 = lambda: _add_op(), fn2 = lambda: _set_op())
            grads = [tf.cond(tf.less(self.mini_flag[0], 0.5), fn1 = lambda: _set_op(gv[0], mgv[0]), fn2 = lambda: _add_op(gv[0], mgv[0]))
                     for (gv, mgv) in zip(self.grads_and_vars, minibatch_grads)]

            with tf.control_dependencies(grads):
                self.mini_flag = tf.assign(self.mini_flag, tf.constant([1], dtype = tf.float32))
        else:
            grads = [mgv[0] for mgv in minibatch_grads]
        grads = [(only_grad, gv[1])
                 for (gv, only_grad) in zip(self.grads_and_vars, grads)]
        return self.mini_flag, grads

    def apply_gradients(self, grads_and_vars, global_step=None):
        """Apply gradients to model variables specified in `grads_and_vars`.

        `apply_gradients` returns an op that calls
        `tf.train.Optimizer.apply_gradients` and then zeros the gradient
        variables stored in `self.grads_and_vars`.

        Args:
            grads_and_vars (list): Description.
            global_step (None, optional): tensorflow global_step variable.

        Returns:
            (tf.Operation): Applies gradient update to model followed by an
                internal gradient zeroing operation to `self.grads_and_vars`.

        """
        self.mini_flag = tf.assign(self.mini_flag, tf.constant([0], dtype = tf.float32))
        # grads_and_vars = self.aggregate_gradients(grads_and_vars, method='average')
        with tf.control_dependencies([self.mini_flag]):
            optimize = self._optimizer.apply_gradients(grads_and_vars,
                                                       global_step=global_step)
        #return [optimize, self.zero_grad()]
        return optimize

    def zero_grad(self):
        if self.grads_and_vars is None:
            self.grads_and_vars = [(
                tf.Variable(tf.zeros_like(var), dtype=tf.float32, trainable=False),
                var) for var in self.var_list]
        return [tf.assign(gv[0], tf.zeros_like(gv[0]))
                for gv in self.grads_and_vars]

    def minimize(self, loss, global_step):
        train_vars = None
        if self.trainable_names is not None:
            log.info('All trainable vars:\n'+str([var.name for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
            train_vars = []
            for scope_name in self.trainable_names:
                new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
                if len(new_vars) == 0:
                    raise ValueError('The scope name, {}, you specified does not contain any trainable variables.'.format(scope_name))
                train_vars.extend(new_vars)
            log.info('Variables to be trained:\n'+str([var.name for var in train_vars]))
        grads_and_vars = self.compute_gradients(loss, var_list=train_vars)
        return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)
