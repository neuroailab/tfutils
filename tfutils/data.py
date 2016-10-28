from __future__ import absolute_import, division, print_function

import threading
import numpy as np
import h5py
import tensorflow as tf


class HDF5DataProvider(object):
    def __init__(self,
                 hdf5source,
                 sourcelist,
                 batch_size,
                 subslice=None,
                 mini_batch_size=None,
                 preprocess=None,
                 postprocess=None,
                 pad=False):

        self.hdf5source = hdf5source
        self.file = h5py.File(self.hdf5source, 'r')
        self.sourcelist = sourcelist
        self.subslice = subslice
        self.subsliceinds = None
        self.preprocess = {} if preprocess is None else preprocess
        self.postprocess = {} if postprocess is None else postprocess

        self.data = {}
        self.sizes = {}
        for source in self.sourcelist:
            self.data[source] = self.file[source]
            if source in self.preprocess:
                print('Preprocessing %s...' % source)
                self.data[source] = self.preprocess[source](self.data[source])
                print('...done')

        for source in sourcelist:
            if self.subslice is None:
                self.sizes[source] = self.data[source].shape
            else:
                if self.subsliceinds is None:
                    if isinstance(self.subslice, str):
                        self.subsliceinds = self.file[self.subslice][:]
                    elif hasattr(self.subslice, '__call__'):
                        self.subsliceinds = self.subslice(self.file, self.sourcelist)
                    elif len(self.subslice) == self.data[source].shape[0]:
                        self.subsliceinds = self.subslice[:]
                    else:
                        self.subsliceinds = np.zeros(self.data[source].shape[0]).astype(np.bool)
                        self.subsliceinds[self.subslice] = True
                    self.subsliceinds = self.subsliceinds.nonzero()[0].astype(int)
                sz = self.data[source].shape
                self.sizes[source] = (self.subsliceinds.shape[0],) + sz[1:]
            if not hasattr(self, 'data_length'):
                self.data_length = self.sizes[source][0]
            assert self.sizes[source][0] == self.data_length, (self.sizes[source], self.data_length)

        self.batch_size = batch_size
        if mini_batch_size is None:
            mini_batch_size = self.batch_size
        self.mini_batch_size = mini_batch_size
        self.total_batches = self.data_length // self.batch_size + 1
        self.curr_batch_num = 0
        self.curr_epoch = 1
        self.pad = pad

    def set_epoch_batch(self, epoch, batch_num):
        self.curr_epoch = epoch
        self.curr_batch_num = batch_num

    def get_next_batch(self):
        data = self.get_batch(self.curr_batch_num)
        self.increment_batch_num()
        return data

    def __iter__(self):
        return self

    def next(self):
        return self.get_next_batch()

    def increment_batch_num(self):
        m = self.total_batches
        if (self.curr_batch_num >= m - 1):
            self.curr_epoch += 1
        self.curr_batch_num = (self.curr_batch_num + 1) % m

    def get_batch(self, cbn):
        data = {}
        startv = cbn * self.batch_size
        endv = (cbn + 1) * self.batch_size
        if self.pad and endv > self.data_length:
            startv = self.data_length - self.batch_size
            endv = startv + self.batch_size
        sourcelist = self.sourcelist
        for source in sourcelist:
            data[source] = self.get_data(self.data[source], slice(startv, endv))
            if source in self.postprocess:
                data[source] = self.postprocess[source](data[source], self.file)
        return data

    def get_data(self, dsource, sliceval):
        if self.subslice is None:
            return dsource[sliceval]
        else:
            subslice_inds = self.subsliceinds[sliceval]
            mbs = self.mini_batch_size
            bn0 = subslice_inds.min() // mbs
            bn1 = subslice_inds.max() // mbs
            stims = []
            for _bn in range(bn0, bn1 + 1):
                _s = np.asarray(dsource[_bn * mbs: (_bn + 1) * mbs])
                new_inds = isin(np.arange(_bn * mbs, (_bn + 1) * mbs), subslice_inds)
                new_array = _s[new_inds]
                stims.append(new_array)
            stims = np.concatenate(stims)
            return stims


def get_unique_labels(larray):
    larray = larray[:]
    labels_unique = np.unique(larray)
    s = larray.argsort()
    cat_s = larray[s]
    ss = np.array([0] + ((cat_s[1:] != cat_s[:-1]).nonzero()[0] + 1).tolist() + [len(cat_s)])
    ssd = ss[1:] - ss[:-1]
    labels = np.repeat(np.arange(len(labels_unique)), ssd)
    larray = labels[perminverse(s)]
    return larray.astype(np.int64)


def perminverse(s):
    """
    Fast inverse of a (numpy) permutation.

    From yamutils
    """
    X = np.array(range(len(s)))
    X[s] = range(len(s))
    return X


def isin(X,Y):
    """
    Indices of elements in a numpy array that appear in another.
    Fast routine for determining indices of elements in numpy array `X` that
    appear in numpy array `Y`, returning a boolean array `Z` such that::
            Z[i] = X[i] in Y
    **Parameters**
            **X** :  numpy array
                    Numpy array to comapare to numpy array `Y`.  For each
                    element of `X`, ask if it is in `Y`.
            **Y** :  numpy array
                    Numpy array to which numpy array `X` is compared.  For each
                    element of `X`, ask if it is in `Y`.
    **Returns**
            **b** :  numpy array (bool)
                    Boolean numpy array, `len(b) = len(X)`.
    **See Also:**
            :func:`tabular.fast.recarrayisin`,
            :func:`tabular.fast.arraydifference`
    """
    if len(Y) > 0:
        T = Y.copy()
        T.sort()
        D = T.searchsorted(X)
        T = np.append(T,np.array([0]))
        W = (T[D] == X)
        if isinstance(W,bool):
            return np.zeros((len(X),),bool)
        else:
            return (T[D] == X)
    else:
        return np.zeros((len(X),),bool)


class CustomQueue(object):

    def __init__(self, node, data_iter, queue_batch_size=128, n_threads=4, seed=0):
        """
        A generic queue for reading data

        Based on https://indico.io/blog/tensorflow-data-input-part2-extensions/
        """
        self.node = node
        self.data_iter = data_iter
        self.queue_batch_size = queue_batch_size
        self.n_threads = n_threads
        self._continue = True

        dtypes = [d.dtype for d in node.values()]
        shapes = [d.get_shape() for d in node.values()]
        self.queue = tf.RandomShuffleQueue(capacity=n_threads * queue_batch_size * 2,
                                        min_after_dequeue=n_threads * queue_batch_size,
                                        dtypes=dtypes,
                                        shapes=shapes,
                                        seed=seed)
        self.enqueue_op = self.queue.enqueue(node.values())
        data_batch = self.queue.dequeue_many(queue_batch_size)
        self.batch = {k:v for k,v in zip(node.keys(), data_batch)}

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for batch in self.data_iter:
            if self._continue:
                sess.run(self.enqueue_op, feed_dict=batch)

    def start_threads(self, sess):
        """ Start background threads to feed queue """
        threads = []
        for n in range(self.n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class ImageNet(HDF5DataProvider):

    def __init__(self, 
                 data_path, 
                 subslice=None,
                 crop_size=None,
                 batch_size=256, 
                 *args, 
                 **kwargs):
        """
        A specific reader for IamgeNet stored as a HDF5 file

        Args:
            - data_path: path to imagenet data
            - subslice: np array for training or eval slice
            - crop_size: for center crop (crop_size x crop_size)
            - *args: extra arguments for HDF5DataProvider
        Kwargs:
            - **kwargs: extra keyword arguments for HDF5DataProvider
        """
        HDF5DataProvider.__init__(self,
            data_path,
            ['data', 'labels'],
            batch_size=1,  # filll up the queue one image at a time
            subslice=subslice,
            preprocess={'labels': get_unique_labels},
            postprocess={'data': self.postproc},
            pad=True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size
        self.node = {'data': tf.placeholder(tf.float32,
                                            shape=(self.crop_size, self.crop_size, 3),
                                            name='data'),
                     'labels': tf.placeholder(tf.int64,
                                              shape=[],
                                              name='labels')}
 
    def postproc(self, ims, f):
        norm = ims / 255. - .5
        resh = norm.reshape((3, 256, 256))
        sw = resh.swapaxes(0, 1).swapaxes(1, 2)
        off = np.random.randint(0, 256 - self.crop_size, size=2)
        images_batch = sw[off[0]: off[0] + self.crop_size,
                          off[1]: off[1] + self.crop_size]
        return images_batch.astype(np.float32)

    def next(self):
        batch = super(ImageNet, self).next()
        feed_dict = {self.node['data']: batch['data'],
                     self.node['labels']: batch['labels'][0]}
        return feed_dict
