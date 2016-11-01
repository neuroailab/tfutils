from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime

import numpy as np
import tensorflow as tf

from tfutils import base, data, model

host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    DATA_PATH = '/om/user/qbilius/imagenet/data.raw'
else:  # agents
    DATA_PATH = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'


class ClipOptimizer(object):
   
    def __init__(self, optimizer_class, *optimizer_args, **optimizer_kwargs):
        self._optimizer = optimizer_class(*optimizer_args, **optimizer_kwargs)

    def compute_gradients(self, *args, **kwargs):
        gvs = self._optimizer.compute_gradients(*args, **kwargs)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                          for grad, var in gvs if grad is not None]
        return capped_gvs

    def minimize(self, loss, global_step):
        grads_and_vars = self.compute_gradients(loss)
        return self._optimizer.apply_gradients(grads_and_vars, 
                                               global_step=global_step)
        
def get_loss(inputs, 
             outputs, 
             target, 
             loss_func, 
             agg_func=None, 
             loss_func_kwargs=None, 
             agg_func_kwargs=None):
    if loss_func_kwargs is None:
        loss_func_kwargs = {}
    loss = loss_func(outputs, inputs[target], **loss_func_kwargs)
    if agg_func is not None:
        if agg_func_kwargs is None:
            agg_func_kwargs = {}
        loss = agg_func(loss, **agg_func_kwargs)
    return loss


def main():
    num_batches_per_epoch = 2**20//256
    params = {}
    params['model_func'] = model.alexnet_nonorm
    params['model_kwargs'] = {}
    params['train_data_func'] = data.ImageNet
    params['train_data_kwargs'] = {'data_path': DATA_PATH,
                                   'crop_size': 224}
    params['loss_func'] = get_loss
    params['loss_kwargs'] = {'target': 'labels',
                             'loss_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
                             'agg_func': tf.reduce_mean}
    params['learning_rate_func'] = tf.train.exponential_decay
    params['learning_rate_kwargs'] = {'learning_rate': 0.01,
                           'decay_steps': num_batches_per_epoch,
                           'decay_rate': 0.95,
                           'staircase': True}
    params['optimizer_func'] = ClipOptimizer
    params['optimizer_kwargs'] = {'optimizer_class': tf.train.MomentumOptimizer,
                                  'momentum': 0.9}
    params['saver_kwargs'] = {'host': 'localhost',
                              'port': 31001,
                              'dbname': 'tfutils-test',
                              'collname': 'test',
                              'exp_id': 'tfutils-test-1'}
    params['queue_kwargs'] = {'seed': 0}
    params['num_steps'] = 10

    return base.run_base(**params)

if __name__ == '__main__':
    main()
