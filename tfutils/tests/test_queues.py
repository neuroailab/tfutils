import time, threading
import numpy as np
import tensorflow as tf


class Queue(object):

    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.node = {'data': tf.placeholder(tf.float32,
                                            shape=[self.batch_size],
                                            name='data')}
        # node = tf.Variable(tf.random_normal([self.batch_size], dtype=tf.float32, stddev=1e-1))
        self.queue = tf.FIFOQueue(capacity=256, dtypes=tf.float32, shapes=[], names=['data'])
        self.enqueue_op = self.queue.enqueue_many(self.node)
        self.batch = self.queue.dequeue_many(self.batch_size)
        self._continue = True

    def loop(self, sess, coord):
        # Thread body: loop until the coordinator indicates a stop was requested.
        # If some condition becomes true, ask the coordinator to stop.
        batch = {self.node['data']: np.random.randn(self.batch_size).astype(np.float32)}
        while not coord.should_stop():
            sess.run(self.enqueue_op, feed_dict=batch)
            if not self._continue:
                print('coord requested stop')
                coord.request_stop()


def main():
    queue = Queue()
    target = tf.reduce_mean(queue.batch['data'])
    # Main code: create a coordinator.
    coord = tf.train.Coordinator()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Create 1 thread that run 'loop()'
    threads = [threading.Thread(target=queue.loop, args=(sess, coord,)) for i in xrange(1)]

    # Start the threads and wait for all of them to stop.
    for t in threads: t.start()

    sess.run(target)

    queue._continue = False
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    print('batch 1')
    main()
    print('batch 2')
    main()