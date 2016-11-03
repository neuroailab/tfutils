from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime

import numpy as np
import tensorflow as tf

from tfutils import base, data, model

host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    # DATA_PATH = '/mindhive/dicarlolab/common/imagenet/data.raw'
    DATA_PATH = '/om/user/qbilius/imagenet/data.raw'
else:  # agents
    DATA_PATH = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'


class DataInMem(object):

    def __init__(self, batch_size=256, *args, **kwargs):
        self.batch_size = batch_size
        self.batch = {'data': tf.Variable(tf.random_normal([self.batch_size, 224, 224, 3],
                                            dtype=tf.float32, stddev=1e-1)),
                    'labels': tf.Variable(tf.ones([self.batch_size], dtype=tf.int32))}

class Data(object):

    def __init__(self, batch_size=256, *args, **kwargs):
        self.batch_size = batch_size
        self.node = {'data': tf.placeholder(tf.float32,
                                          shape=(self.batch_size, 224, 224, 3)),
                     'labels': tf.placeholder(tf.int64, shape=[self.batch_size])}
        self._data = np.random.uniform(-.5, .5, size=[self.batch_size, 224, 224, 3])
        self._labels = np.random.randint(0, 1000, size=self.batch_size)

    def __iter__(self):
        return self

    def next(self):
        feed_dict = {self.node['data']: self._data.astype(np.float32),
                     self.node['labels']: self._labels.astype(np.int64)}
        return feed_dict


class DataQueue(data.CustomQueue):

    def __init__(self, batch_size=256, n_threads=4, *args, **kwargs):
        node = {'data': tf.placeholder(tf.float32,
                                          shape=(224, 224, 3)),
                     'labels': tf.placeholder(tf.int64, shape=[])}
        super(DataQueue, self).__init__(node, self, queue_batch_size=batch_size,
                             n_threads=n_threads)
        self._data = np.random.uniform(-.5, .5, size=[224, 224, 3])
        self._labels = np.random.randint(0, 1000)

    def __iter__(self):
        return self

    def next(self):
        feed_dict = {self.node['data']: self._data.astype(np.float32),
                     self.node['labels']: self._labels}
        return feed_dict


def timeit(sess, data, train_targets, nsteps=100, burn_in=10):
    """
    Args:
        - queues (~ data)
        - saver
        - targets
    """
    # initialize and/or restore variables for graph
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    # start our custom queue runner's threads
    if hasattr(data, 'start_threads'):
        data.start_threads(sess)
        batch_size = data.queue_batch_size
    else:
        batch_size = data.batch_size

    dur = []
    for step in xrange(0, burn_in + nsteps):
        start_time = time.time()
        if hasattr(data, 'start_threads') or not hasattr(data, 'next'):
            sess.run(train_targets)
        else:
            sess.run(train_targets, feed_dict=data.next())
        end_time = time.time()
        d = end_time - start_time
        if step > burn_in:
            dur.append(1000 * d)
            print('{}: {:.0f} ms'.format(step, 1000 * d))

    print('Forward-backward across {} steps (batch size = {}): {:.0f} +/- {:.0f} msec / batch'.format(nsteps, batch_size, np.mean(dur), np.std(dur)))
    sess.close()
    return dur


def main(data):
    if not hasattr(data, 'batch'):
        batch = data.node
    else:
        batch = data.batch
    outputs, _ = model.alexnet_nonorm(batch['data'])
    targets = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, batch['labels']))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    dur = timeit(sess, data, targets, nsteps=20)
    return dur


if __name__ == '__main__':
    base.get_params()

    durs = []
    print('test 0...')
    durs.append(['Data input using feed_dict', main(DataInMem())])
    print('test 1...')
    durs.append(['Numpy data input using feed_dict', main(Data())])
    print('test 2...')
    queue = DataQueue()
    durs.append(['Numpy data input using queues', main(queue)])
    queue._continue = False  # TODO: cleaning close threads
    print('test 3....')
    imagenet = data.ImageNet(DATA_PATH, crop_size=224)
    imagenet = data.CustomQueue(imagenet.node, imagenet)
    durs.append(['HDF5 data input using queues', main(imagenet)])
    time.sleep(2)  # clear queue outputs
    print()
    print('{:-^80}'.format('Benchmarking results'))
    for message, dur in durs:
        print('-', message + ': {:.0f} +/- {:.0f} msec / batch'.format(np.mean(dur), np.std(dur)))
