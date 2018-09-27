from __future__ import division, print_function, absolute_import
import os
import sys

import numpy as np
import tensorflow as tf
import argparse

sys.path.insert(0, '.')
sys.path.insert(0, '..')
from tfutils import base, optimizer, model_tool
from tfutils.utils import online_agg
from imagenet_data import dataset_func


DATA_LEN_IMAGENET_FULL = 1281167
VALIDATION_LEN = 50000


def get_learning_rate(
        global_step, 
        nb_per_epoch,
        init_lr,
        ):
    """
    Drop by 10 for every 30 epochs
    """
    curr_epoch = tf.div(
            tf.cast(global_step, tf.float32), 
            tf.cast(nb_per_epoch, tf.float32))
    drop_times = tf.cast(tf.minimum(curr_epoch / 30, 3), tf.int32)
    drop_times = tf.cast(drop_times, tf.float32)
    drop_times = tf.Print(drop_times, [drop_times, curr_epoch], message='Drop times')
    curr_lr = init_lr * tf.pow(0.1, drop_times)
    return curr_lr


def rep_loss_func(
        inputs,
        output,
        **kwargs
        ):
    return {
            'loss_pure': tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=output, 
                    labels=inputs['labels'])
                ),
            }


def loss_and_in_top_k(inputs, outputs, target):
    return {
            'loss': tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=outputs, 
                labels=inputs[target]),
            'top1': tf.nn.in_top_k(outputs, inputs[target], 1),
            'top5': tf.nn.in_top_k(outputs, inputs[target], 5)}


def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss)\
            + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))


def get_parser():
    parser = argparse.ArgumentParser(
            description='Train AlexNet using tfutils')
    parser.add_argument(
            '--image_dir',
            default=None, required=True,
            type=str, action='store', help='Where the tfrecords are stored')
    parser.add_argument(
            '--gpu',
            default='0', type=str, action='store', 
            help='Availabel GPUs, multiple GPUs are separated by ","')
    parser.add_argument(
            '--batch_size',
            default=256,
            type=int, action='store', help='Batch size')
    return parser


def get_params_from_arg(args):
    """
    This function gets parameters needed for tfutils.train_from_params()
    """
    multi_gpu = len(args.gpu.split(','))
    dbname = 'tfutils_tutorial'
    collname = 'example'
    exp_id = 'alexnet_ctl'
    NUM_BATCHES_PER_EPOCH = DATA_LEN_IMAGENET_FULL // args.batch_size 

    # save_params: defining where to save the models
    save_params = {
            'host': 'localhost',
            'port': 27009,
            'dbname': dbname,
            'collname': collname,
            'exp_id': exp_id,
            'do_save': True,
            'save_metrics_freq': 1000,
            'save_valid_freq': NUM_BATCHES_PER_EPOCH,
            'save_filters_freq': NUM_BATCHES_PER_EPOCH,
            'cache_filters_freq': NUM_BATCHES_PER_EPOCH,
            'cache_dir': None, # where local model caches will be stored
            }

    # load_params: defining where to load, if needed
    load_params = {
            'host': 'localhost',
            'port': 27009,
            'dbname': dbname,
            'collname': collname,
            'exp_id': exp_id,
            'do_restore': True,
            'query': None,
            }

    # model_params: a function that will build the model
    model_params = {
            'func': model_tool.alexnet_tfutils,
            'norm': False,
            'seed': 0,
            }
    if multi_gpu > 1:
        # How to use multiple gpu training:
        model_params['num_gpus'] = multi_gpu
        model_params['devices'] = \
                ['/gpu:%i' % idx for idx in range(multi_gpu)]

    # train_params: parameters about training data
    data_param_base = {
            'func': dataset_func,
            'image_dir': args.image_dir,
            }
    train_data_param = {
            'is_train': True,
            'batch_size': args.batch_size
            }
    train_data_param.update(data_param_base)
    train_params = {
            'validate_first': True, # You may want to turn this off at debugging 
            'data_params': train_data_param,
            'queue_params': None,
            'thres_loss': float('Inf'),
            'num_steps': 120 * NUM_BATCHES_PER_EPOCH,
            }
    ## Add other loss reports (loss_model, loss_noise)
    train_params['targets'] = {
            'func': rep_loss_func,
            }

    # loss_params: parameters to build the loss
    loss_params = {
            'pred_targets': ['labels'],
            'agg_func': mean_loss_with_reg,
            'loss_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
            }

    # learning_rate_params: build the learning rate
    # For now, just stay the same
    learning_rate_params = {
            'func': get_learning_rate,
            'init_lr': 0.01,
            'nb_per_epoch': NUM_BATCHES_PER_EPOCH,
            }

    # optimizer_params: use tfutils optimizer,
    # as mini batch is implemented there
    optimizer_params = {
            'optimizer': tf.train.MomentumOptimizer,
            'momentum': .9,
            }

    # validation_params: control the validation
    ## Basic parameters for both validation on train and val
    val_step_num = int(VALIDATION_LEN / args.batch_size)
    val_param_base = {
        'targets': {
            'func': loss_and_in_top_k,
            'target': 'labels',
            },
        'num_steps': val_step_num,
        'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        'online_agg_func': online_agg,
        }
    val_data_param_base = {
            'is_train': False,
            'q_cap': args.batch_size,
            'batch_size': args.batch_size,
            }
    ## Validation on validation set
    topn_val_data_param = {
            'file_pattern': 'validation-*',
            }
    topn_val_data_param.update(data_param_base)
    topn_val_data_param.update(val_data_param_base)
    topn_val_param = {
        'data_params': topn_val_data_param,
        }
    topn_val_param.update(val_param_base)
    ## Validation on training set
    topn_train_data_param = {
            'file_pattern': 'train-*',
            }
    topn_train_data_param.update(data_param_base)
    topn_train_data_param.update(val_data_param_base)
    topn_train_param = {
        'data_params': topn_train_data_param,
        }
    topn_train_param.update(val_param_base)
    validation_params = {
            'topn_val': topn_val_param,
            'topn_train': topn_train_param,
            }

    # Put all parameters together
    params = {
            'save_params': save_params,
            'load_params': load_params,
            'model_params': model_params,
            'train_params': train_params,
            'loss_params': loss_params,
            'learning_rate_params': learning_rate_params,
            'optimizer_params': optimizer_params,
            'validation_params': validation_params,
            'skip_check': True,
            }
    return params


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Get params needed, start training
    params = get_params_from_arg(args)
    base.train_from_params(**params)


if __name__ == '__main__':
    main()
