from __future__ import division, print_function, absolute_import
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from tfutils import base, data, model, optimizer


host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    DATA_PATH = '/mnt/fs0/datasets/imagenet2012_tf'
elif 'gpu-1' in host or 'gpu-2' in host:
    DATA_PATH = '/scratch/imagenet2012_tf'
elif host.startswith('braintree'):
    DATA_PATH = '/home/qbilius/data/imagenet2012_tf'
else:
    DATA_PATH = '/home/qbilius/data/imagenet2012_tf'


def loss_and_in_top_k(inputs, outputs, target):
    return {'loss': tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputs[target]),
            'top1': tf.nn.in_top_k(outputs, inputs[target], 1),
            'top5': tf.nn.in_top_k(outputs, inputs[target], 5)}


def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res


def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


def exponential_decay(global_step,
                      learning_rate=.01,
                      decay_factor=.95,
                      decay_steps=1,
                      ):
    # Decay the learning rate exponentially based on the number of steps.
    if decay_factor is None:
        lr = learning_rate  # just a constant.
    else:
        # Calculate the learning rate schedule.
        lr = tf.train.exponential_decay(learning_rate,  # Base learning rate.
                                        global_step,  # Current index into the dataset.
                                        decay_steps,  # Decay step
                                        decay_factor,  # Decay rate.
                                        staircase=True)
    return lr


BATCH_SIZE = 256
NUM_BATCHES_PER_EPOCH = data.ImageNet.N_TRAIN // BATCH_SIZE
IMAGE_SIZE_CROP = 224

params = {
    'save_params': {
        'host': 'localhost',
        'port': 27017,
        'dbname': 'AlexNet-test',
        'collname': 'alexnet',
        'exp_id': 'trainval0',

        'do_save': False,
        'save_initial_filters': False,
        'save_metrics_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 3000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 30000,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        # 'host': 'localhost',
        # 'port': 29101,
        # 'dbname': 'alexnet-test',
        # 'collname': 'alexnet',
        # 'exp_id': 'trainval0',
        'do_restore': False,
        'query': None
    },

    'model_params': {
        'func': model.alexnet_tfutils,
        'seed': 0
    },

    'train_params': {
        'data_params': {
            'func': data.ImageNetTF,
            'source_dirs': [DATA_PATH],
            'crop_size': IMAGE_SIZE_CROP,
            'batch_size': BATCH_SIZE,
            'file_pattern': 'train*.tfrecords',
            'n_threads': 4
        },
        'queue_params': {
            'queue_type': 'fifo',
            'batch_size': BATCH_SIZE,
            'capacity': None,
            'min_after_dequeue': None,
            'seed': 0,
        },
        'thres_loss': 1000,
        'num_steps': 90 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'validate_first': True,
    },

    'loss_params': {
        'targets': 'labels',
        'agg_func': mean_loss_with_reg,
        'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': .01,
        'decay_rate': .95,
        'decay_steps': NUM_BATCHES_PER_EPOCH,  # exponential decay each epoch
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.MomentumOptimizer,
        'clip': True,
        'momentum': .9
    },

    'validation_params': {
        'topn_val': {
            'data_params': {
                'func': data.ImageNetTF,
                'source_dirs': [DATA_PATH],
                'crop_size': IMAGE_SIZE_CROP,
                'batch_size': BATCH_SIZE,
                'file_pattern': 'val*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_and_in_top_k,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'capacity': None,
                'min_after_dequeue': None,
                'seed': 0,
            },
            'num_steps': data.ImageNet.N_VAL // BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
        'topn_train': {
            'data_params': {
                'func': data.ImageNetTF,
                'source_dirs': [DATA_PATH],
                'crop_size': IMAGE_SIZE_CROP,
                'batch_size': BATCH_SIZE,
                'file_pattern': 'trainval*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_and_in_top_k,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': BATCH_SIZE,
                'capacity': None,
                'min_after_dequeue': None,
                'seed': 0,
            },
            'num_steps': data.ImageNet.N_VAL // BATCH_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        }
    },
    'log_device_placement': False,  # if variable placement has to be logged
}


if __name__ == '__main__':
    base.get_params()
    base.train_from_params(**params)
