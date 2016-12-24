"""
The is the basic illustration of training.
"""
from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tfutils import base, model, utils

class MNIST(object):
    def __init__(self, batch_size=100, group='train'):
        """
        A specific reader for IamgeNet stored as a HDF5 file

        Args:
            - data_path: path to imagenet data
            - crop_size: for center crop (crop_size x crop_size)
            - *args: extra arguments for HDF5DataProvider
        Kwargs:
            - **kwargs: extra keyword arguments for HDF5DataProvider
        """
        self.batch_size = batch_size
        data = read_data_sets('/tmp')
        if group == 'train':
            self.data = data.train
        elif group == 'test':
            self.data = data.train
        elif group == 'validation':
            self.data = data.train
        else:
            raise ValueError('MNIST data input "{}" not known'.format(group))

        self.node = {'images': tf.placeholder(tf.float32,
                                              shape=(self.batch_size, 784),
                                              name='images'),
                     'labels': tf.placeholder(tf.int32,
                                              shape=[self.batch_size],
                                              name='labels')}

    def __iter__(self):
        return self

    def next(self):
        batch = self.data.next_batch(self.batch_size)
        feed_dict = {self.node['images']: batch[0],
                     self.node['labels']: batch[1]}
        return feed_dict


def test1():
    num_batches_per_epoch = 10000//256
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['train_params'] = {'data': {'func': MNIST,
                                       'batch_size': 100,
                                       'group': 'train'
                                      },
                              'queue_params': {'queue_type': 'fifo',
                                               'batch_size': 100,
                                               'n_threads': 4}}
    params['learning_rate_params'] = {'learning_rate': 0.05,
                                      'decay_steps': num_batches_per_epoch,
                                      'decay_rate': 0.95,
                                      'staircase': True}
    params['save_params'] = {'host': 'localhost',
                             'port': 31001,
                             'dbname': 'tfutils-test',
                             'collname': 'test',
                             'exp_id': 'tfutils-test-7',
                             'save_valid_freq': 20,
                             'save_filters_freq': 200,
                             'cache_filters_freq': 100}
    params['num_steps'] = 1030

    base.train_from_params(**params)

    #params['save_params']['do_save'] = False
    #base.train_from_params(**params)


def get_targets(inputs, outputs, **params):
    #names = [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]
    #print("NAMES", names)
    f = tf.get_default_graph().get_tensor_by_name('validation/test/hidden1/fc:0')
    targets = {'loss': utils.get_loss(inputs, outputs, **params),
               'features': f}
    return targets


def test2():
    num_batches_per_epoch = 2**20//256
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['validation_params'] = {'test': {'data': {'func': MNIST,
                                                     'batch_size': 100,
                                                     'group': 'train'
                                                 },
                                            'queue_params': {'queue_type': 'fifo',
                                                             'batch_size': 100,
                                                             'n_threads': 4},
                                            'num_steps': 10,
                                            'agg_func': utils.mean_dict}}
    params['load_params'] = {'host': 'localhost',
                             'port': 31001,
                             'dbname': 'tfutils-test',
                             'collname': 'test',
                             'exp_id': 'tfutils-test-7'}
    params['save_params'] = {'exp_id': 'tfutils-test-7-valid'}

    base.test_from_params(**params)


    targdict = {'func': get_targets}
    targdict.update(base.default_loss_params())
    params['validation_params']['test']['targets'] = targdict
    params['validation_params']['test'].pop('agg_func')
    params['validation_params']['test']['online_agg_func'] = utils.reduce_mean_dict
    params['save_params']['save_to_gfs'] = ('features',)
    base.test_from_params(**params)


if __name__ == '__main__':
    test1()
