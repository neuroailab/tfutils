from __future__ import absolute_import, division, print_function

import os
import functools
import itertools
import copy
import cPickle

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tfutils.utils import isstring

class ParallelByFileProviderBase(object):
    def __init__(self,
                 source_paths,
                 meta_dict,
                 trans_dict=None,
                 batch_size=256,
                 n_threads=4,
                 postprocess=None,
                 args=None,
                 shuffle=False,
                 shuffle_seed=0):
        """
        - source (str): path where file(s) reside
        - sourcedict (dict of tf.dtypes): dict of datatypes where the keys are 
          the keys in the tfrecords file to use as source dataarrays and the values are the tensorflow datatypes
        - batch_size (int, default=256): size of batches to be returned
        - n_threads (int, default=4): number of threads to be used
        """
        
        if not isinstance(source_paths, list):
            assert isstring(source_paths)
            source_paths = [source_paths]
        self.source_paths = source_paths
        self.n_attrs = len(self.source_paths)
        self.meta_dict = meta_dict
        self.trans_dict = trans_dict
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.postprocess = {} if postprocess is None else postprocess
        self.args = args
        for source in self.postprocess:
            assert source in self.meta_dict.keys(), 'postprocess has to be a subset of meta_dict'

        self.datasources = self.get_data_paths(source_paths)
        assert len(self.datasources) == self.n_attrs

        if self.n_attrs == 1:            
            fq = tf.train.string_input_producer(self.datasources[0],
                                                shuffle=shuffle,
                                                seed=shuffle_seed)
            self.file_queues = [[fq]] * self.n_threads
        else:
            self.file_queues = []
            tuples = zip(*self.datasources)
            if shuffle:
                rng = np.random.RandomState(seed=shuffle_seed)
                tuples = random_cycle(tuples, rng)
            else:
                tuples = itertools.cycle(tuples)
            for n in range(self.n_threads):
                coord = Coordinator(tuples)
                fqs = []
                for j in range(self.n_attrs):
                    item = Item(coord, j)
                    func = tf.py_func(item.next, [], [tf.string])
                    fq = tf.train.string_input_producer(func)
                    fqs.append(fq)
                self.file_queues.append(fqs)

    def init_threads(self):
        self.input_ops = []
        for thread_num in range(self.n_threads):
            op = {}
            for attr_num in range(self.n_attrs):
                fq = self.file_queues[thread_num][attr_num]
                if self.args is not None:
                    args = self.args[attr_num]
                else:
                    args = ()
                _op = self.get_input_op(fq, *args)
                if self.trans_dict:
                    sp = self.source_paths[attr_num]
                    for (sp0, k) in self.trans_dict:
                        if sp == sp0 and k in _op:
                            _op[self.trans_dict[(sp0, k)]] = _op.pop(k)
                op.update(_op)
            self.input_ops.append(op)
        self.apply_postprocessing()            
        return self.input_ops

    def get_data_paths(self, paths):
        datasources = []
        for path in paths:
            if os.path.isdir(path):
                tfrecord_pattern = os.path.join(path, '*.tfrecords')
                datasource = tf.gfile.Glob(tfrecord_pattern)
                datasource.sort()
                datasources.append(datasource)
            else:
                datasources.append([path])
        dl = map(len, datasources)
        assert all([dl[0] == d for d in dl[1:]]), dl
        return datasources

    def get_input_op(self):
        raise NotImplementedError()

    def apply_postprocessing(self):
        ops = self.input_ops
        for i in range(len(ops)):
            for source in self.postprocess:
                op = ops[i][source]
                for func, args, kwargs in self.postprocess[source]:
                    op = func(op, *args, **kwargs)
                ops[i][source] = op


def get_parser(dtype):
    dtype = dtype if dtype in [tf.float32, tf.int64] else tf.string
    return tf.FixedLenFeature([], dtype)


def parse_standard_tfmeta(paths, trans_dict=None):
    d = {}
    parser_list = []
    for path in paths:
        mpath = os.path.join(path, 'meta.pkl')
        if os.path.isfile(mpath):
            d0 = cPickle.load(open(mpath))
            parsers = {k: get_parser(d0[k]['dtype']) for k in d0}
            parser_list.append(parsers)
            if trans_dict:
                for k in d0:
                    if (path, k) in trans_dict:
                        d0[trans_dict[(path, k)]] = d0.pop(k)
            assert set(d0.keys()).intersection(d.keys()) == set(), (d0.keys(), d.keys())
            d.update(d0)
    return d, parser_list
            

class TFRecordsDataProvider(ParallelByFileProviderBase):
    def __init__(self,
                 source_paths,
                 meta_dict=None,
                 trans_dict=None,
                 postprocess=None,
                 batch_size=256,
                 **kwargs):
        """
        """
        parsed_meta_dict, parser_list = parse_standard_tfmeta(source_paths, trans_dict=trans_dict)
        if meta_dict is None:
            meta_dict = parsed_meta_dict
        else:
            for k in meta_dict:
                assert k in parsed_meta_dict
                for kv in parsed_meta_dict[k]:
                    if kv not in meta_dict[k]:
                        meta_dict[k][kv] = parsed_meta_dict[k][kv]

        if postprocess is None:
            postprocess = {}
        for k in meta_dict:
            if k not in postprocess:
                postprocess[k] = []
            dtype = meta_dict[k]['dtype']
            if dtype not in [tf.float32, tf.int64]:
                postprocess[k].insert(0, (tf.decode_raw, (meta_dict[k]['dtype'], ), {}))
                postprocess[k].insert(1, (tf.reshape, ([-1] + meta_dict[k]['shape'], ), {}))

        super(TFRecordsDataProvider, self).__init__(source_paths,
                                                    meta_dict,
                                                    trans_dict=trans_dict,
                                                    args=[(p, ) for p in parser_list],
                                                    batch_size=batch_size,
                                                    postprocess=postprocess,
                                                    **kwargs)

        
    def get_input_op(self, fq, parsers):
        reader = tf.TFRecordReader()
        _, serialized_data = reader.read_up_to(fq, self.batch_size)
        return tf.parse_example(serialized_data, parsers)
            

class ParallelBySliceProvider(object):
    def __init__(self,
                 basefunc,
                 kwargs,
                 mode='block',
                 batch_size=256,
                 n_threads=1):
        self.func = basefunc
        self.kwargs = kwargs
        self.mode = mode
        self.n_threads = n_threads
        self.batch_size = batch_size

    def init_threads(self):
        n = self.n_threads
        ops = []
        tester = self.func(batch_size=self.batch_size, **self.kwargs)
        N = tester.data_length
        labels = tester.labels

        if self.mode == 'block':
            blocksize = N / n
            ends = [[i * blocksize, (i+1) * blocksize] for i in range(n)]
            ends[-1][1] = max(ends[-1][1], N)
            subslices = [np.arange(e0, e1).astype(np.int) for e0, e1 in ends]
        elif self.mode == 'alternate':
            subslices = [np.arange(N)[i::].astype(np.int) for i in range(n)]

        testbatch = tester.next()
        for n in range(self.n_threads):
            kwargs = copy.deepcopy(self.kwargs)
            if 'subslice' not in self.kwargs:
                kwargs['subslice'] = subslices[n]
            else:
                good_inds = tester.subsliceinds
                kwargs['subslice'] = good_inds[subslices[n]]
            kwargs['batch_size'] = self.batch_size
            dp = self.func(**kwargs)
            op = tf.py_func(dp.next, [], [t.dtype for t in testbatch])
            for _op, t in zip(op, testbatch):
                _op.set_shape(t.shape)
            op = dict(zip(labels, op))
            ops.append(op)
        return ops
    
        
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
        - postprocess (dict of callables): functions for postprocess data.  Keys of this are subset of sourcelist.
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
                        self.subsliceinds = self.subslice[:].astype(np.int)
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
        self.total_batches = (self.data_length - 1) // self.batch_size + 1
        self.curr_batch_num = 0
        self.curr_epoch = 1
        self.pad = pad

    @property
    def labels(self):
        return self.sourcelist

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
        b = self.get_next_batch()
        return [b[k] for k in self.sourcelist]

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


def isin(X, Y):
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
        T = np.append(T, np.array([0]))
        W = (T[D] == X)
        if isinstance(W, bool):
            return np.zeros((len(X), ), bool)
        else:
            return (T[D] == X)
    else:
        return np.zeros((len(X), ), bool)


def get_queue(nodes,
              queue_type='fifo',
              batch_size=256,
              capacity=None,
              seed=0):
    """ A generic queue for reading data
        Built on top of https://indico.io/blog/tensorflow-data-input-part2-extensions/
    """
    if capacity is None:
        capacity = 2 * batch_size
        
    names = []
    dtypes = []
    shapes = []

    for name in nodes.keys():
        names.append(name)
        dtypes.append(nodes[name].dtype)
        shapes.append(nodes[name].get_shape()[1: ])

    if queue_type == 'random':
        queue = tf.RandomShuffleQueue(capacity=capacity,
                                           min_after_dequeue=capacity // 2,
                                           dtypes=dtypes,
                                           shapes=shapes,
                                           names=names,
                                           seed=seed)
    elif queue_type == 'fifo':
        queue = tf.FIFOQueue(capacity=capacity,
                                  dtypes=dtypes,
                                  shapes=shapes,
                                  names=names)
    elif queue_type == 'padding_fifo':
        queue = tf.PaddingFIFOQueue(capacity=capacity,
                                         dtypes=dtypes,
                                         shapes=shapes,
                                         names=names)
    elif queue_type == 'priority':
        queue = tf.PriorityQueue(capacity=capacity,
                                      types=dtypes,
                                      shapes=shapes,
                                      names=names)
    else:
        Exception('Queue type %s not recognized' % queue_type)

    return queue


class MNIST(object):

    def __init__(self,
                 data_path=None,
                 group='train',
                 batch_size=100,
                 n_threads=1):
        """
        Kwargs:
            - data_path: path to imagenet data
            - group: train, validation, test
            - batch_size
        """
        self.n_threads = n_threads
        self.batch_size = batch_size

        if data_path is None:
            data_path = '/tmp'
        data = read_data_sets(data_path)

        if group == 'train':
            self.data = data.train
        elif group == 'test':
            self.data = data.test
        elif group == 'validation':
            self.data = data.validation
        else:
            raise ValueError('MNIST data input "{}" not known'.format(group))

    def __iter__(self):
        return self

    def next(self):
        batch = self.data.next_batch(self.batch_size)
        feed_dict = {'images': batch[0], 'labels': batch[1].astype(np.int32)}
        return feed_dict

    def init_threads(self):
        func = functools.partial(self.data.next_batch, self.batch_size)
        batches = [tf.py_func(func, [], [tf.float32, tf.uint8]) for _ in range(self.n_threads)]
        for b in batches:
            b[0].set_shape([self.batch_size, 784])
            b[1].set_shape([self.batch_size])
        ops = [{'images': b[0], 'labels': tf.cast(b[1], tf.int32)} for b in batches]
        return ops


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
        images_key = group + '/images'
        labels_key = group + '/labels'
        super(ImageNet, self).__init__(
            data_path,
            [images_key, labels_key],
            batch_size=batch_size,
            postprocess={images_key: self.postproc_img, labels_key: self.postproc_labels},
            pad=True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

    def postproc_img(self, ims, f):
        norm = ims.astype(np.float32) / 255.0

        off = np.random.randint(0, 256 - self.crop_size, size=2)

        if self.group=='train':
            off = np.random.randint(0, 256 - self.crop_size, size=2)
        else:
            off = int((256 - self.crop_size)/2)
            off = [off, off]
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]
        return images_batch

    def postproc_labels(self, labels, f):
        return labels.astype(np.int32)

    # def next(self):
    #     batch = super(ImageNet, self).next()
    #     feed_dict = {'images': np.squeeze(batch[self.group + '/images']),
    #                  'labels': np.squeeze(batch[self.group + '/labels'])}
    #     return feed_dict


class Coordinator(object):
    def __init__(self, itr):
        self.curval = {}
        self.itr = itr

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


def random_cycle(ls, rng):
    local_ls = ls[:] # defensive copy
    while True:
        rng.shuffle(local_ls)
        for e in local_ls:
            yield e
