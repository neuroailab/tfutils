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


def get_optimizer(loss, learning_rate, optimizer_func, grad_clip=True, optimizer_kwargs={}):
    optimizer = optimizer_func(learning_rate=learning_rate, **optimizer_kwargs)
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
    
    

if __name__ == '__main__':
    num_batches_per_epoch = 2**20//256
    params = {}
    params['model_func'] = model.alexnet_nonorm
    params['model_kwargs'] = {}
    params['train_data_func'] = data.ImageNet
    params['train_data_kwargs'] = {'data_path': DATA_PATH,
                                   'crop_size': 224}
    params['loss_func'] = tf.nn.sparse_softmax_cross_entropy_with_logits
    params['loss_func_kwargs'] = {}
    params['lr_func'] = exponential_decay
    params['lr_kwargs'] = {'learning_rate': 0.01,
                           'decay_steps': num_batches_per_epoch,
                           'decay_rate': 0.95,
                           'stair_case': True}
    params['opt_func'] = get_optimizer
    params['opt_kwargs'] = {'optimizer_func': tf.train.MomentumOptimizer,
                            'optimizer_kwargs': {'momentum': 0.9}}
    params['saver_kwargs'] = {'host': 'localhost',
                              'port': 31001,
                              'dbname': 'tfutils-test',
                              'collname': 'test',
                              'exp_id': 'tfutils-test-0'}


    base.run_base(**params)
