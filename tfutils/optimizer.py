import tensorflow as tf


class ClipOptimizer(object):

    def __init__(self, optimizer_class, clip=True, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)
        self.clip = clip

    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        if self.clip:
            # gradient clipping. Some gradients returned are 'None' because
            # no relation between the variable and loss; so we skip those.
            gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                   for grad, var in gvs if grad is not None]
        return gvs

    def minimize(self, loss, global_step):
        grads_and_vars = self.compute_gradients(loss)
        return self._optimizer.apply_gradients(grads_and_vars,
                                               global_step=global_step)
