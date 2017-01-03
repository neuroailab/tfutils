from __future__ import division, print_function, absolute_import
import os
import copy
import time
import tempfile
import sys

import h5py
import tqdm
import numpy as np
import pandas
import tensorflow as tf

from tfutils import base, data, model

sys.path.insert(0, '..')

BATCH_SIZE = 256
NSTEPS = 20
IMSIZE = 224


def create_hdf5(n_img, path=None, shape=(IMSIZE, IMSIZE, 3), im_dtype=np.float32):
    if path is None:
        path = '/tmp'
    # os.makedirs(path)
    tempf = tempfile.NamedTemporaryFile(suffix='.hdf5', dir=path, delete=False)
    tempf.close()
    with h5py.File(tempf.name, 'w') as f:
        f.create_dataset('data', ((n_img, ) + shape), dtype=im_dtype)
        f.create_dataset('labels', (n_img, ), dtype=np.int64)
        img = np.random.randn(*shape)
        label = np.random.randint(1000)
        for i in tqdm.trange(n_img, desc='creating hdf5 file'):
            f['data'][i] = img
            f['labels'][i] = label
    return tempf.name


class DataInMem(object):

    def __init__(self, batch_size=256, *args, **kwargs):
        self.kind = 'in gpu memory'
        self.batch_size = batch_size
        self.node = {'data': tf.Variable(tf.random_normal([self.batch_size, IMSIZE, IMSIZE, 3],
                                                          dtype=tf.float32, stddev=1e-1)),
                     'labels': tf.Variable(tf.zeros([self.batch_size], dtype=tf.int32))}
        self.batch = self.node


class DataNoRead(object):

    def __init__(self, batch_size=256, *args, **kwargs):
        self.kind = 'in ram (feed_dict)'
        self.batch_size = batch_size
        # if self.batch_size == 1:
        #     self.node = {'data': tf.placeholder(tf.float32,
        #                                          shape=(IMSIZE, IMSIZE, 3)),
        #                  'labels': tf.placeholder(tf.int64, shape=[])}
        # else:
        #     self.node = {'data': tf.placeholder(tf.float32,
        #                                          shape=(self.batch_size, IMSIZE, IMSIZE, 3)),
        #                  'labels': tf.placeholder(tf.int64, shape=[self.batch_size])}

        self._data = np.random.uniform(-.5, .5, size=[self.batch_size, IMSIZE, IMSIZE, 3])
        self._labels = np.random.randint(0, 1000, size=[self.batch_size])
        # self.batch = self.node

    def __iter__(self):
        return self

    def next(self):
        feed_dict = {'data': np.squeeze(self._data.astype(np.float32)),
                     'labels': np.squeeze(self._labels.astype(np.int64))}
        return feed_dict


class DataHDF5(data.HDF5DataProvider):

    def __init__(self, path=None, batch_size=256, queue_batch_size=256,
                 dtype=np.float32, postproc='identity'):
        self.kind = 'hdf5 read (feed_dict)'
        self.batch_size = batch_size
        self.queue_batch_size = queue_batch_size
        self.data_path = create_hdf5(self.queue_batch_size * NSTEPS,
                                     path=path,
                                     shape=(IMSIZE, IMSIZE, 3),
                                     im_dtype=dtype)
        postproc_func = getattr(self, postproc)

        super(DataHDF5, self).__init__(self.data_path,
                                       ['data', 'labels'],
                                       batch_size=self.batch_size,
                                       postprocess={'data': postproc_func},
                                       pad=True)

    def identity(self, ims, f):
        return ims

    def convert_dtype(self, ims, f):
        return ims.astype(np.float32)

    def copy(self, ims, f):
        return np.copy(ims)

    def next(self):
        batch = super(DataHDF5, self).next()
        feed_dict = {'data': np.squeeze(batch['data']),
                     'labels': np.squeeze(batch['labels'])}
        return feed_dict

    def cleanup(self):
        os.remove(self.data_path)


def time_hdf5():
    data_path = create_hdf5(BATCH_SIZE * NSTEPS)

    f = h5py.File(data_path)
    durs = []
    for step in tqdm.trange(NSTEPS, desc='running hdf5'):
        start_time = time.time()
        arr = f['data'][BATCH_SIZE * step: BATCH_SIZE * (step+1)]
        read_time = time.time()
        arr = copy.deepcopy(arr)
        copy_time = time.time()
        durs.append(['hdf5 read', step, read_time - start_time])
        durs.append(['hdf5 copy', step, copy_time - read_time])
    f.close()
    os.remove(data_path)
    durs = pandas.DataFrame(durs, columns=['kind', 'stepno', 'dur'])
    return durs


def time_tf(data):
    m = model.alexnet_nonorm(data.batch['data'])
    targets = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(m.output, data.batch['labels']))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # start our custom queue runner's threads
    if hasattr(data, 'start_threads'):
        data.start_threads(sess)

    durs = []
    for step in tqdm.trange(NSTEPS, desc='running ' + data.kind):
        start_time = time.time()
        if hasattr(data, 'start_threads') or not hasattr(data, 'next'):
            sess.run(targets)
        else:
            batch = data.next()
            feed_dict = {node: batch[name] for name, node in data.batch.items()}
            sess.run(targets, feed_dict=feed_dict)
        end_time = time.time()
        durs.append([data.kind, step, end_time - start_time])

    if hasattr(data, 'stop_threads'):
        data.stop_threads(sess)

    sess.close()

    durs = pandas.DataFrame(durs, columns=['kind', 'stepno', 'dur'])
    return durs


def hdf5_tests():
    df = time_hdf5()
    df.kind = df.kind.astype('category', ordered=True, categories=df.kind.unique())
    gr = df.groupby('kind').dur.agg([np.mean, np.median, np.std])
    print(gr)


def standard_tests():
    df = []

    tf.reset_default_graph()
    durs = time_tf(DataInMem())
    df.append(durs)

    tf.reset_default_graph()
    d = DataNoRead()
    d.batch = {}
    d.batch['data'] = tf.placeholder(np.float32, shape=(BATCH_SIZE, IMSIZE, IMSIZE, 3), name='data')
    d.batch['labels'] = tf.placeholder(np.int64, shape=[BATCH_SIZE], name='labels')
    durs = time_tf(d)
    df.append(durs)

    tf.reset_default_graph()
    d = DataHDF5()
    d.batch = {}
    d.batch['data'] = tf.placeholder(np.float32, shape=(BATCH_SIZE, IMSIZE, IMSIZE, 3), name='data')
    d.batch['labels'] = tf.placeholder(np.int64, shape=[BATCH_SIZE], name='labels')
    durs = time_tf(d)
    df.append(durs)

    tf.reset_default_graph()
    d = DataHDF5(batch_size=1)
    queue = data.Queue(d, queue_type='fifo', batch_size=BATCH_SIZE)
    queue.kind = 'hdf5 read (queue)'
    durs = time_tf(queue)
    d.cleanup()
    df.append(durs)

    tf.reset_default_graph()
    d = DataHDF5(batch_size=1, postproc='copy')
    queue = data.Queue(d, queue_type='fifo', batch_size=BATCH_SIZE)
    queue.kind = 'hdf5 read+copy (queue)'
    durs = time_tf(queue)
    d.cleanup()
    df.append(durs)

    tf.reset_default_graph()
    d = DataHDF5(batch_size=1, dtype=np.uint8, postproc='convert_dtype')
    queue = data.Queue(d, queue_type='fifo', batch_size=BATCH_SIZE)
    queue.kind = 'hdf5 convert dtype (queue)'
    durs = time_tf(queue)
    d.cleanup()
    df.append(durs)

    # Print results
    df = pandas.concat(df, ignore_index=True)
    df.kind = df.kind.astype('category', ordered=True, categories=df.kind.unique())
    gr = df.groupby('kind').dur.agg([np.mean, np.median, np.std])
    print(gr)


def search_queue_params():
    df = []

    data_batch_sizes = np.logspace(0, 8, num=9, base=2, dtype=int)
    capacities = np.logspace(0, 12, num=13, base=2, dtype=int)
    nthreads = np.logspace(0, 5, num=6, base=2, dtype=int)

    for nth in nthreads:
        for data_batch_size in data_batch_sizes:
            for capacity in capacities:
                cap = nth * capacity

                tf.reset_default_graph()
                d = DataHDF5(batch_size=data_batch_size)
                queue = data.Queue(d.node, d,
                                   queue_type='fifo',
                                   batch_size=BATCH_SIZE,
                                   capacity=cap,
                                   n_threads=nth)
                queue.kind = '{} / {} / {}'.format(nth, data_batch_size, capacity)
                durs = time_tf(queue)
                durs['data batch size'] = data_batch_size
                durs['queue capacity'] = cap
                durs['nthreads'] = nth
                df.append(durs)
                d.cleanup()

    df = pandas.concat(df, ignore_index=True)
    df.kind = df.kind.astype('category', ordered=True, categories=df.kind.unique())
    df.to_pickle('/home/qbilius/mh17/computed/search_queue_params.pkl')
    print(df.groupby(['nthreads', 'data batch size', 'queue capacity']).dur.mean())


if __name__ == '__main__':
    base.get_params()
    # hdf5_tests()
    standard_tests()
    # search_queue_params()
