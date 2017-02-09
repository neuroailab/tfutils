import functools
import threading


class Q(object):
    def __init__(self, tuples):
        assert all([len(t) == len(tuples[0]) for t in tuples[1:]])
        self.t = len(tuples[0])
        self.tuples = iter(tuples)
        self.block = None

    def next(self, name, j):
        while (self.block != name and self.block is not None):
            pass
        
        if self.block is None:
            self.block = name
            try:
                self.curval = self.tuples.next()
            except StopIteration:
                self.block = None
                raise StopIteration
            self.remains = range(len(self.curval))
            
        rval = self.curval[j]
        self.remains.remove(j)
        if len(self.remains) == 0:
            self.block = None
            
        return rval

import time
import numpy as np

gorp = []
rng = np.random.RandomState(0)

def func(qv, name):
    while True:
        rval = []
        try:
            for j in range(qv.t):
                r = qv.next(name, j)
                time.sleep(rng.uniform())
                rval.append(r)
            gorp.append([rval, name])
        except StopIteration:
            return


if __name__ == '__main__':
    tuples = [('a%d' % i, 'b%d' % i) for i in range(10)]
    qv = Q(tuples)
    names = ['x', 'y']
    threads = [threading.Thread(target=functools.partial(func, qv, name)) for name in names]
    for t in threads:
        t.daemon = True
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(gorp)
