from __future__ import division, print_function, absolute_import

import os
import tempfile

from numpy.testing import assert_equal
import numpy as np
import h5py
import tensorflow as tf

from tfutils import data
from tfutils import base


def create_hdf5(total_size):
    tempf = tempfile.NamedTemporaryFile(suffix='.hdf5', dir='/tmp', delete=False)
    tempf.close()
    with h5py.File(tempf.name, 'w') as f:
        f.create_dataset('data', (total_size, ), dtype=np.float)
        f.create_dataset('inds', (total_size, ), dtype=np.int64)
        f['data'][:] = np.sin(np.arange(total_size))
        f['inds'][:] = np.arange(total_size)
    return tempf.name


def test_fifo_one_thread():
    batch_size = 100
    data_batch_size = 20
    total_size = batch_size * 10

    tmp_path = create_hdf5(total_size)

    dp = data.ParallelBySliceProvider(basefunc=data.HDF5DataReader,
                                      kwargs={'hdf5source': tmp_path,
                                              'sourcelist': ['data', 'inds']},
                                      batch_size=batch_size,
                                      n_threads=1)

    ops = dp.init_ops()
    queue = base.get_queue(ops[0], queue_type='fifo')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))

    inputs = queue.dequeue_many(data_batch_size)

    sess = tf.Session()
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    coord, threads = base.start_queues(sess)

    r = sess.run(inputs)
    r1 = sess.run(inputs)

    base.stop_queues(sess, [queue], coord, threads)
    sess.close()

    os.remove(tmp_path)

    assert_equal(r['data'], np.sin(np.arange(20)))
    assert_equal(r['inds'], np.arange(20))

    assert_equal(r1['data'], np.sin(np.arange(20, 40)))
    assert_equal(r1['inds'], np.arange(20, 40))


def test_fifo_two_threads_ops():
    batch_size = 100
    total_size = batch_size * 10

    tmp_path = create_hdf5(total_size)

    dp = data.ParallelBySliceProvider(basefunc=data.HDF5DataReader,
                                      kwargs={'hdf5source': tmp_path,
                                              'sourcelist': ['data', 'inds']},
                                      batch_size=batch_size,
                                      n_threads=2)

    ops = dp.init_ops()

    sess = tf.Session()

    r = sess.run(ops)
    r1 = sess.run(ops)

    sess.close()
    os.remove(tmp_path)

    inds0 = map(np.array, [_r['inds'] for _r in r])
    inds1 = map(np.array, [_r['inds'] for _r in r1])
    assert_equal(inds0[0], np.arange(100))
    assert_equal(inds0[1], np.arange(500, 600))
    assert_equal(inds1[0], np.arange(100, 200))
    assert_equal(inds1[1], np.arange(600, 700))


def test_fifo_four_threads():
    batch_size = 100
    data_batch_size = 20
    total_size = batch_size * 10

    tmp_path = create_hdf5(total_size)

    dp = data.ParallelBySliceProvider(basefunc=data.HDF5DataReader,
                                      kwargs={'hdf5source': tmp_path,
                                              'sourcelist': ['data', 'inds']},
                                      batch_size=batch_size,
                                      n_threads=4)

    ops = dp.init_ops()
    queue = base.get_queue(ops[0], queue_type='fifo')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))

    inputs = queue.dequeue_many(data_batch_size)

    sess = tf.Session()
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    coord, threads = base.start_queues(sess)

    out = []
    for _ in range(100):
        out.append(sess.run(inputs))

    base.stop_queues(sess, [queue], coord, threads)
    sess.close()

    os.remove(tmp_path)

    seen_inds = np.unique(np.concatenate([o['inds'] for o in out]))

    assert_equal(seen_inds, np.arange(1000))


def random_four_threads():
    batch_size = 100
    data_batch_size = 200
    total_size = batch_size * 10

    tmp_path = create_hdf5(total_size)

    dp = data.ParallelBySliceProvider(basefunc=data.HDF5DataReader,
                                      kwargs={'hdf5source': tmp_path,
                                              'sourcelist': ['data', 'inds']},
                                      batch_size=batch_size,
                                      n_threads=4)

    ops = dp.init_ops()
    queue = base.get_queue(ops[0], queue_type='random')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))

    inputs = queue.dequeue_many(data_batch_size)

    sess = tf.Session()
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    coord, threads = base.start_queues(sess)

    out = []
    for _ in range(3):
        out.append(sess.run(inputs))

    base.stop_queues(sess, [queue], coord, threads)
    sess.close()

    os.remove(tmp_path)

    out_inds = [o['inds'] for o in out]
    seen_inds = np.unique(np.concatenate(out_inds))

    return len(seen_inds), seen_inds.max()


def test_random_four_threads():
    sl = []
    m = []
    for ind in range(50):
        print(ind)
        a, b = random_four_threads()
        sl.append(a)
        m.append(b)
    sl = np.array(sl)
    m = np.array(m)
    assert sl.min() > 400
    assert m.min() > 740
    return sl, m
