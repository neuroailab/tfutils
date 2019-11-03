from tensorflow.python.framework import ops
from tensorflow.python.ops.losses import losses
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.contrib.tpu.python.ops import tpu_ops

class MultiCrossShardOptimizer(tpu_optimizer.CrossShardOptimizer):
    def __init__(self,
                 opt,
                 reduction=losses.Reduction.MEAN,
                 name='MultiCrossShardOptimizer',
                 group_assignment=None):
        super(MultiCrossShardOptimizer, self).__init__(opt,
                                           reduction=reduction,
                                           name=name,
                                           group_assignment=group_assignment)

    def compute_gradients(self, loss, var_list=None, **kwargs):
        """ This is adapted from:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/tpu/python/tpu/tpu_optimizer.py#L100
        loss is a list of lists of outer length num_optimizers.
        Therefore, for each optimizer's loss, we multiply each loss by the
        scale
        """
        num_shards = tpu_function.get_tpu_context().number_of_shards
        if num_shards is None:
            logging.warning(
                    "CrossShardOptimizer should be used within a tpu_shard_context, but "
                    "got unset number_of_shards. Assuming 1.")
            num_shards = 1

        subgroup_size = self._verify_and_get_subgroup_size(self._group_assignment,
                                                           num_shards)

        scaled_losses= []
        for loss_idx, curr_loss in enumerate(loss):
            if num_shards > 1 and self._reduction == losses.Reduction.MEAN:
                if self._group_assignment:
                    scale = 1.0 / subgroup_size
                else:
                    scale = 1.0 / num_shards
                curr_loss *= scale
            scaled_losses.insert(loss_idx, curr_loss)

        return self._opt.compute_gradients(scaled_losses, var_list=var_list, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """ This is adapted from:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/tpu/python/tpu/tpu_optimizer.py#L142
        The intention here is to deal with the case of having multiple
        optimizers, wherein grads_and_vars is a list of lists of outer length
        num_optimizers.
        Therefore, for each optimizer's grads and vars, we compute the
        cross_replica_sum of those gradients across replicas.
        """

        summed_grads_and_vars = []
        for opt_idx, curr_gv in enumerate(grads_and_vars):
            curr_summed_grads_and_vars = []
            for (grad, var) in curr_gv:
                if grad is None:
                    curr_summed_grads_and_vars.append((grad, var))
                else:
                    with ops.colocate_with(grad):
                        curr_summed_grads_and_vars.append((tpu_ops.cross_replica_sum(
                                grad, self._group_assignment), var))

            summed_grads_and_vars.insert(opt_idx, curr_summed_grads_and_vars)

        return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)

    def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=1, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
        """This is adapted from:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/optimizer.py#L355
        The intention here is to deal with the general case of having multipler
        optimizers, wherein grads_and_vars is a list of lists of outer length
        num_optimizers.
        NOTE: gate_gradients is set to its default value of 1
        (https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/optimizer.py#L310)
        """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        for opt_idx, curr_gv in enumerate(grads_and_vars):
            curr_gv_wo_none = [v for g, v in curr_gv if g is not None]
            if not curr_gv_wo_none:
              raise ValueError(
                  "No gradients provided for any variable, check your graph for ops"
                  " that do not support gradients, between variables %s and loss %s." %
                  ([str(v) for _, v in curr_gv], loss[opt_idx]))

        return self.apply_gradients(grads_and_vars, global_step=global_step,
                                    name=name)
