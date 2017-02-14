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


def test1():
    dp = d.TFRecordsParallelByFileProvider(source_paths,
                                           trans_dicts=trans_dicts,
                                           n_threads=2,
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

    for i in range(100):
        res = sess.run(inputs)
        assert res['images'].shape == (20, 32, 32, 3)
        assert_allclose(res['images'].mean(1).mean(1).mean(1), res['means'], rtol=1e-05)
        assert_equal(res['ids'], res['ids1'])


def test2():
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
    inputs = queue.dequeue_many(31)

    testlist = np.arange(3100) % 1600
    for i in range(100):
        res = sess.run(inputs)
        assert res['images'].shape == (31, 32, 32, 3)
        assert_allclose(res['images'].mean(1).mean(1).mean(1), res['means'], rtol=1e-05)
        assert_equal(res['ids'], res['ids1'])
        assert_equal(res['ids'], testlist[31 * i: 31 * (i+1)])
