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

        """
        - hdf5source (str): path where hdf5 file resides
        - sourcelist (list of strs): list of keys in the hdf5file to use as source dataarrays
        - batch_size (int): size of batches to be returned
        - subslice (string, array of ints, callable):
             if str: name of key in hdf5file refering to array of indexes into the source dataarrays
             if array of ints: indexes into the source dataarrays
             if callable: function producing array of indexes into the source datarrays
           Regardless of how it's constructed, the provider subsets its returns to the only the indices
           specified in the subslice.
        - mini_batch_size (int):  Used only if subslice is specifiied, this sets the size of minibatches used
          when constructing one full batch within the subslice to return
        - preprocess (dict of callables): functions for preprocessing data in the datasources.  keys of this are subset
          of sourcelist. preprocessing is done on object instantiation
        - postprocess (dict of callables): functions for postprocess data.  Keys of this are subset of sourcelist.
          Postprocessing is done when get_batch is called.
        - pad (bool): whether to pad data returned if amount of data left to return is less then full batch size
        """
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


class Queue(object):
    """ A generic queue for reading data
        Based on https://indico.io/blog/tensorflow-data-input-part2-extensions/
    """

    def __init__(self,
                 node,
                 data_iter,
                 queue_type='fifo',
                 batch_size=256,
                 n_threads=4,
                 seed=0):
        self.node = node
        self.data_iter = data_iter
        self.batch_size = batch_size
        self.n_threads = n_threads
        self._continue = True

        dtypes = [d.dtype for d in node.values()]
        if self.data_iter.batch_size > 1:
            shapes = [d.get_shape()[1:] for d in node.values()]
        else:
            shapes = [d.get_shape() for d in node.values()]

        if queue_type == 'random':
            self.queue = tf.RandomShuffleQueue(capacity=n_threads * batch_size * 2,
                                               min_after_dequeue=n_threads * batch_size,
                                               dtypes=dtypes,
                                               shapes=shapes,
                                               names=node.keys(),
                                               seed=seed)
        elif queue_type == 'fifo':
            self.queue = tf.FIFOQueue(capacity=n_threads * batch_size * 2,
                                      dtypes=dtypes,
                                      shapes=shapes,
                                      names=node.keys())
        elif queue_type == 'padding_fifo':
            self.queue = tf.PaddingFIFOQueue(capacity=n_threads * batch_size * 2,
                                             dtypes=dtypes,
                                             shapes=shapes,
                                             names=node.keys())
        elif queue_type == 'priority':
            self.queue = tf.PriorityQueue(capacity=n_threads * batch_size * 2,
                                          types=dtypes,
                                          shapes=shapes,
                                          names=node.keys())
        else:
            Exception('Queue type %s not recognized' % queue_type)

        if self.data_iter.batch_size > 1:
            self.enqueue_op = self.queue.enqueue_many(node)
        else:
            self.enqueue_op = self.queue.enqueue(node)
        self.batch = self.queue.dequeue_many(batch_size)

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

    N_TRAIN = 1281167
    N_VAL = 50000
    N_TRAIN_VAL = 50000

    def __init__(self,
                 data_path,
                 group='train',
                 batch_size=1,
                 crop_size=None,
                 *args,
                 **kwargs):
        """
        A specific reader for ImageNet stored as a HDF5 file

        Args:
            - data_path
                path to imagenet data
        Kwargs:
            - group (str, default: 'train')
                Which subset of the ImageNet set you want: train, val, train_val.
                The latter contains 50k images from the train set (50 per category),
                so that you can directly compare performance on the validation set
                to the performance on the train set to track overfitting.
            - batch_size (int, default: 1)
                Number of images to return when `next` is called. By default set
                to 1 since it is expected to be used with queues where reading one
                image at a time is ok.
            - crop_size (int or None, default: None)
                For center crop (crop_size x crop_size). If None, no cropping will occur.
            - *args, **kwargs
                Extra arguments for HDF5DataProvider
        """
        self.group = group
        images = group + '/images'
        labels = group + '/labels'
        HDF5DataProvider.__init__(self,
            data_path,
            [images, labels],
            batch_size=batch_size,
            postprocess={images: self.postproc_img},
            pad=True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        if self.batch_size == 1:
            self.node = {'images': tf.placeholder(tf.float32,
                                    shape=(self.crop_size, self.crop_size, 3),
                                    name='images'),
                        'labels': tf.placeholder(tf.int64,
                                                shape=[],
                                                name='labels')}
        else:
            self.node = {'images': tf.placeholder(tf.float32,
                                    shape=(self.batch_size, self.crop_size, self.crop_size, 3),
                                    name='images'),
                        'labels': tf.placeholder(tf.int64,
                                                 shape=(self.batch_size, ),
                                                 name='labels')}

    def postproc_img(self, ims, f):
        norm = ims.astype(np.float32) / 255
        off = np.random.randint(0, 256 - self.crop_size, size=2)
        if self.batch_size == 1:
            images_batch = norm[off[0]: off[0] + self.crop_size,
                                off[1]: off[1] + self.crop_size]
        else:
            images_batch = norm[:,
                                off[0]: off[0] + self.crop_size,
                                off[1]: off[1] + self.crop_size]
        return images_batch

    def next(self):
        batch = super(ImageNet, self).next()
        feed_dict = {self.node['images']: np.squeeze(batch[self.group + '/images']),
                     self.node['labels']: np.squeeze(batch[self.group + '/labels'])}
        return feed_dict