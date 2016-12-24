from __future__ import division, print_function, absolute_import
import tempfile
import unittest

import h5py
import numpy as np

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tfutils import base, model, data


def create_hdf5(n_img, path=None, shape=(224, 224, 3)):
    if path is None:
        path = '/tmp'
    tempf = tempfile.NamedTemporaryFile(suffix='.hdf5', dir=path, delete=False)
    tempf.close()
    with h5py.File(tempf.name, 'w') as f:
        f.create_dataset('images', ((n_img, ) + shape), dtype=np.float32)
        f.create_dataset('labels', (n_img, ), dtype=np.int64)
        img = np.random.randn(*shape)
        label = np.random.randint(1000)
        for i in range(n_img):
            f['images'][i] = img
            f['labels'][i] = label
    return tempf.name


class DataHDF5(data.HDF5DataProvider):

    def __init__(self,
                 data_path=None,
                 crop_size=None,
                 *args,
                 **kwargs):
        """
        A specific reader for IamgeNet stored as a HDF5 file

        Args:
            - data_path: path to imagenet data
            - crop_size: for center crop (crop_size x crop_size)
            - *args: extra arguments for HDF5DataProvider
        Kwargs:
            - **kwargs: extra keyword arguments for HDF5DataProvider
        """
        super(DataHDF5, self).__init__(
            data_path,
            ['images', 'labels'],
            batch_size=1,  # fill up the queue one image at a time
            pad=True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size
        self.node = {'images': tf.placeholder(tf.float32,
                                            shape=(self.crop_size, self.crop_size, 3),
                                            name='images'),
                     'labels': tf.placeholder(tf.int64,
                                              shape=[],
                                              name='labels')}

    def next(self):
        batch = super(DataHDF5, self).next()
        feed_dict = {self.node['images']: batch['images'][0],
                     self.node['labels']: batch['labels'][0]}
        return feed_dict


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


class TestRun(unittest.TestCase):

    def basic_setup(self, num_steps=100):
        params = {'model_params': {'func': model.mnist_tfutils},
                  'train_params': {'data': {'func': MNIST,
                                            'batch_size': 100,
                                            'group': 'train'},
                                   'queue_params': {'queue_type': 'fifo',
                                                    'batch_size': 100,
                                                    'n_threads': 4}},
                  'learning_rate_params': {'learning_rate': 0.01,
                                           'decay_steps': 1,
                                           'decay_rate': 0.95,
                                           'staircase': True},
                  'save_params': {'host': 'localhost',
                                  'port': 31001,
                                  'dbname': 'tfutils-test',
                                  'collname': 'test',
                                  'exp_id': 'tfutils-test-7',
                                  'save_valid_freq': num_steps // 10,
                                  'save_filters_freq': num_steps // 2,
                                  'cache_filters_freq': num_steps // 4},
                  'load_params': {'do_restore': True},
                  'num_steps': num_steps}
        return params

    def test_basic(self):
        params = self.basic_setup()
        return base.train_from_params(**params)

    def test_nosave(self):
        params = self.basic_setup()
        params['save_params']['do_save'] = False
        return base.train_from_params(**params)
