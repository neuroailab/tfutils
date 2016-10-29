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


def exponential_decay(**kwargs):
    global_step = [v for v in tf.all_variables() if v.name == 'global_step:0'][0]
    return tf.train.exponential_decay(global_step=global_step,
                                      **kwargs)


def get_optimizer(loss, learning_rate, opt_func, opt_kwargs={}, grad_clip=True):
    optimizer = opt_func(learning_rate=learning_rate, **opt_kwargs)
    gvs = optimizer.compute_gradients(loss)
    global_step = [v for v in tf.all_variables() if v.name == 'global_step:0'][0]
    if grad_clip:
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                      for grad, var in gvs if grad is not None]
        # gradient clipping. Some gradients returned are 'None' because
        # no relation between the variable and tot loss; so we skip those.
        optimizer = optimizer.apply_gradients(capped_gvs,
                                              global_step=global_step)
        print('Gradients clipped')
    else:
        optimizer = optimizer.apply_gradients(gvs, global_step=global_step)
        print('Gradients not clipped')

    return optimizer
    

def get_loss(inputs, outputs, target, loss_func, agg_func=None, loss_func_kwargs=None, agg_func_kwargs=None):
    if loss_func_kwargs is None:
        loss_func_kwargs = {}
    loss = loss_func(outputs, inputs[target], **loss_func_kwargs)
    if agg_func is not None:
        if agg_func_kwargs is None:
            agg_func_kwargs = {}
        loss = agg_func(loss, **agg_func_kwargs)
    return loss


if __name__ == '__main__':
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
    params['learning_rate_func'] = exponential_decay
    params['learning_rate_kwargs'] = {'learning_rate': 0.01,
                           'decay_steps': num_batches_per_epoch,
                           'decay_rate': 0.95,
                           'staircase': True}
    params['optimizer_func'] = get_optimizer
    params['optimizer_kwargs'] = {'opt_func': tf.train.MomentumOptimizer,
                                  'opt_kwargs': {'momentum': 0.9}}
    params['saver_kwargs'] = {'host': 'localhost',
                              'port': 31001,
                              'dbname': 'tfutils-test',
                              'collname': 'test',
                              'exp_id': 'tfutils-test-0'}


    base.run_base(**params)
