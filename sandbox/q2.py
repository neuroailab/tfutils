import tensorflow as tf
import itertools
import numpy as np

def random_cycle(ls, rng):
    local_ls = ls[:] # defensive copy
    while True:
        rng.shuffle(local_ls)
        for e in local_ls:
            yield e

class Coordinator(object):
    def __init__(self, lst, shuffle=False, seed=0):
        self.curval = {}
        self.lst = lst
        self.shuffle = shuffle
        rng = np.random.RandomState(seed=seed)
        if self.shuffle:
            self.itr = random_cycle(lst, rng)
        else:
            self.itr = itertools.cycle(lst)

    def next(self, j):
        if not self.curval:
            curval = self.itr.next()
            if not hasattr(curval, 'keys'):
                n = len(curval)
                curval = {i: curval[i] for i in range(n)}
            self.curval = curval
        return self.curval.pop(j)
        

class Item(object):
    def __init__(self, coordinator, j):
        self.j = j
        self.coordinator = coordinator

    def next(self):
        while True:
            try:
                val = self.coordinator.next(self.j)
            except KeyError:
                pass
            else:
                return val

if __name__ == '__main__':
    sess = tf.Session()
    tuples = [('a%d' % i, 'b%d' % i) for i in range(10)]
    coord = Coordinator(tuples, shuffle=True)
    item = Item(coord, 1)
    func = tf.py_func(item.next, [], [tf.string])
    q = tf.train.string_input_producer(func, shuffle=True)
    item0 = Item(coord, 0)
    func0 = tf.py_func(item0.next, [], [tf.string])
    q0 = tf.train.string_input_producer(func0)

    tf.train.start_queue_runners(sess=sess)
    for i in range(10):
        print(sess.run(q0.dequeue()))
        print(sess.run(q.dequeue()))

