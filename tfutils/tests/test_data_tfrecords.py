import numpy as np
from numpy.testing import assert_allclose, assert_equal
import os
import tfutils.data as d
import tfutils.base as b
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
source_paths = [os.path.join(dir_path, 'tftestdata/images'),
                os.path.join(dir_path, 'tftestdata/means')]
trans_dicts = [None, {'ids': 'ids1'}]


def test_ops():
    """Tests the basic init_ops funcions.
    """
    dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           trans_dicts=trans_dicts,
                                           n_threads=4,
                                           batch_size=20,
                                           shuffle=False)
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)

    N = 1000
    for i in range(N):
        res = sess.run([[fq.dequeue() for fq in fqs] for fqs in dp.file_queues])
        x, y = res[0]
        print('%d of %d' % (i, N))
        assert x.split('/')[-1] == y.split('/')[-1]


def test_four_threads_random_and_shuffle():
    """
    Tests that the provider works when all the "things" are happening,
    e.g. multiple threads, random queue and shuffle file queue.
    """
    dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           trans_dicts=trans_dicts,
                                           n_threads=4,
                                           batch_size=20,
                                           shuffle=True)
    sess = tf.Session()
    ops = dp.init_ops()
    queue = b.get_queue(ops[0], queue_type='random')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    tf.train.start_queue_runners(sess=sess)
    inputs = queue.dequeue_many(20)

    N = 1000
    for i in range(N):
        print('%d of %d' % (i, N))
        res = sess.run(inputs)
        assert res['images'].shape == (20, 32, 32, 3)
        assert_equal(res['ids'], res['ids1'])
        assert_allclose(res['images'].mean(1).mean(1).mean(1), res['means'], rtol=1e-05)


def test_fifo_one_thread_no_shuffle():
    """Tests that when one thread with FIFO queue is used,
    the data comes out in exactly the expected order.
    """
    dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           trans_dicts=trans_dicts,
                                           n_threads=1,
                                           batch_size=20,
                                           shuffle=False)
    sess = tf.Session()
    ops = dp.init_ops()
    queue = b.get_queue(ops[0], queue_type='fifo')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    tf.train.start_queue_runners(sess=sess)
    K = 31
    inputs = queue.dequeue_many(K)
    N = 1000
    testlist = np.arange(K * N) % 1600
    for i in range(N):
        print('%d of %d' % (i, N))
        res = sess.run(inputs)
        assert res['images'].shape == (K, 32, 32, 3)
        assert_allclose(res['images'].mean(1).mean(1).mean(1), res['means'], rtol=1e-05)
        assert_equal(res['ids'], testlist[K * i: K * (i+1)])
        assert_equal(res['ids'], res['ids1'])


def test_postprocess():
    """
    This test uses the data in tftestdata2/ to illustrate how to read out
    something that has been written as a string but is "really" an integer.
    The data in tftestdata2/ids is just a single attribute, namely "ids",
    written out as a string but actually it really represents integers.
    """
    source_paths = [os.path.join(dir_path, 'tftestdata2/ids')]
    postprocess = {'ids': [(tf.string_to_number, (tf.int32, ), {})]}
    dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           n_threads=1,
                                           batch_size=20,
                                           shuffle=False,
                                           postprocess=postprocess)
    sess = tf.Session()
    ops = dp.init_ops()
    queue = b.get_queue(ops[0], queue_type='fifo')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    tf.train.start_queue_runners(sess=sess)
    K = 20
    inputs = queue.dequeue_many(K)
    N = 100
    testlist = np.arange(K * N) % 160
    for i in range(N):
        print('%d of %d' % (i, N))
        res = sess.run(inputs)
        assert_equal(res['ids'], testlist[K * i: K * (i+1)])


def test_item_selection():
    """Tests that the option of subselecting keys by passing meta_dicts
    works.
    """
    meta_dicts = ['ids', ['ids', 'means']]
    dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           meta_dicts=meta_dicts,
                                           trans_dicts=trans_dicts,
                                           n_threads=4,
                                           batch_size=20,
                                           shuffle=True)
    sess = tf.Session()
    ops = dp.init_ops()
    queue = b.get_queue(ops[0], queue_type='random')
    enqueue_ops = []
    for op in ops:
        enqueue_ops.append(queue.enqueue_many(op))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(queue, enqueue_ops))
    tf.train.start_queue_runners(sess=sess)
    inputs = queue.dequeue_many(20)
    N = 1000
    for i in range(N):
        print('%d of %d' % (i, N))
        res = sess.run(inputs)
        assert_equal(res['ids'], res['ids1'])
        assert set(res.keys()) == set(['ids', 'ids1', 'means'])
