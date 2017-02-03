from __future__ import absolute_import, division, print_function

import sys
import threading
import os
import functools

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

class TFRecordsDataProviderBase(object):
    def __init__(self,
                 tfsource,
                 sourcedict,
                 batch_size=256,
                 n_threads=4,
                ):
        """
        - tfsource (str): path where tfrecords file(s) reside
        - sourcedict (dict of tf.dtypes): dict of datatypes where the keys are the keys in the tfrecords file to use as source dataarrays and the values are the tensorflow datatypes
        - batch_size (int, default=256): size of batches to be returned
        - n_threads (int, default=4): number of threads to be used
        """
	self.init(tfsource, sourcedict, batch_size, n_threads)

    def init(self, tfsource, sourcedict, batch_size, num_threads):
        self.num_threads = num_threads
        self.sourcedict = sourcedict
        self.batch_size = batch_size
        self.readers = []

        self.tfsource = tfsource
        self.datasources = self.get_data_paths(tfsource)
        self.filename_queue = tf.train.string_input_producer(self.datasources, \
		shuffle=False) #TODO use number of epochs to control padding?

        self.features = {}
        for source in self.sourcedict:
            self.features[source] = tf.FixedLenFeature([], self.sourcedict[source])

    def init_threads(self):
        self.input_ops = []
        self.dtypes = {}
        self.shapes = None #unconstrained shapes

        for thread in range(self.num_threads):
            reader = self.create_input_provider()
            features = self.get_input_op(reader)
            self.input_ops.append(features)
            if len(self.dtypes.keys()) == 0:
                for k in features:
                    self.dtypes[k] = self.sourcedict[k]
        return [self.input_ops, self.dtypes, self.shapes]

    def get_data_paths(self, path):
        if os.path.isdir(path):
            tfrecord_pattern = os.path.join(path, '*.tfrecords')
            datasources = tf.gfile.Glob(tfrecord_pattern)
            return datasources.sort()
        else:
            return [path]

    def set_batch(self, batch_num):
        self.move_ptr_to(batch_num)

    def move_ptr_to(self, batch_num):
        raise NotImplementedError

    def parse_serialized_data(self, data):
        return tf.parse_example(data, self.features)

    def create_input_provider(self):
        self.readers.append(tf.TFRecordReader())
        return self.readers[-1]

    def get_input_op(self, reader):
        _, serialized_data = reader.read_up_to(self.filename_queue, self.batch_size)
        return self.parse_serialized_data(serialized_data)


class TFRecordsDataProvider(TFRecordsDataProviderBase):
    def __init__(self,
                 tfsource,
                 sourcedict,
                 imagelist=None,
                 batch_size=256,
                 n_threads=4,
                 postprocess=None,
                 py_postprocess=None,
                ):
        """
        - tfsource (str): path where tfrecords file(s) reside
        - sourcedict (dict of tf.dtypes): dict of datatypes where the keys are the keys in the tfrecords file to use as source dataarrays and the values are the tensorflow datatypes
        - imagelist (list of strs): list of image type keys in the tfrecords file that have to be decoded from raw bytes format and reshaped to their original form, e. g. numpy arrays or serialized images, imagelist is a subset of sourcedict
        - batch_size (int, default=256): size of batches to be returned
        - num_threads (int, default=4): number of threads to be used
        - postprocess (dict of callables): functions for postprocess data using tf methods.  Keys of this are subset of sourcedict.
        - py_postprocess (dict of (callable, tf return types) tuples): functions for postprocess data using python methods. Keys of this are subset of sourcedict.
        """
        return self.init_from_base(tfsource, sourcedict, imagelist, \
                batch_size, n_threads, postprocess, py_postprocess)

    def init_from_base(self, tfsource, sourcedict, imagelist, \
            batch_size, num_threads, postprocess, py_postprocess):

        self.imagelist = imagelist
        self.postprocess = {} if postprocess is None else postprocess
        self.py_postprocess = {} if py_postprocess is None else py_postprocess

        for source in self.imagelist:
            assert source in sourcedict.keys(), \
                        'imagelist has to be a subset of sourcedict'
        for source in self.postprocess:
            assert source in sourcedict.keys(), \
                        'postprocess has to be a subset of sourcedict'
        for source in self.py_postprocess:
            assert source in sourcedict.keys(), \
                        'postprocess_py has to be a subset of sourcedict'

        super(TFRecordsDataProvider, self).__init__(
                tfsource,
                sourcedict,
                batch_size,
                num_threads)

    def get_dtypes(self, sources):
        dtypes = {} 
        for source in sources:
            if source in self.imagelist:
                dtypes[source] = tf.uint8
            else:
                dtypes[source] = sources[source]
        return dtypes

    def get_shapes(self, sources, tfsource):
        shapes = {}
        image_shape = self.get_image_shape(tfsource)
        for source in sources:
            if source in self.imagelist:
                shapes[source] = image_shape
            else:
                shapes[source] = []
        return shapes

    def get_image_shape(self, tfsource):
        datasources = super(TFRecordsDataProvider, self).get_data_paths(tfsource)
        assert len(self.datasources) > 0, 'No data files provided!'
        file_ptr = tf.python_io.tf_record_iterator(path=datasources[0])

        data = file_ptr.next()
        example = tf.train.Example()
        example.ParseFromString(data)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        channels = int(example.features.feature['channels'].int64_list.value[0])

        file_ptr.close()
        return [height, width, channels]

    def init_threads(self):
        self.input_ops, self.dtypes, self.shapes = \
                super(TFRecordsDataProvider, self).init_threads()
        if self.imagelist is not None:
        # update data types, shapes and ops if image decoding was requested  
            self.shapes = self.get_shapes(self.sourcedict, self.tfsource)
            self.dtypes = self.get_dtypes(self.sourcedict)
            self.input_ops = self.decode_data_many(self.input_ops)
        if len(self.postprocess) > 0:
            self.input_ops = self.postprocess_many(self.input_ops)
        if len(self.py_postprocess) > 0:
            self.input_ops = self.py_postprocess_many(self.input_ops)
        return [self.input_ops, self.dtypes, self.shapes]

    def postprocess_many(self, data):
        for i in range(len(data)):
            for source in self.postprocess:
                data[i][source], self.dtypes[source], self.shapes[source] = \
                        self.postprocess[source]( \
                                data[i][source], self.dtypes[source], self.shapes[source])
        return data

    def py_postprocess_many(self, data):
        raise NotImplementedError('Not yet tested!')
        for i in range(len(data)):
            for source in self.py_postprocess:
                func = self.py_postprocess[source][0]
                return_type = self.py_postprocess[source][1]
                data[i][source] = tf.py_func(func, data[i][source], return_type)
        return data

    def decode_data_many(self, data):
        for i in range(len(data)):
            data[i] = self.decode_features(data[i])
        return data

    def decode_features(self, features):
        for k in features:
            if k in self.imagelist:
                features[k] = tf.decode_raw(features[k], tf.uint8)
                features[k] = tf.reshape(features[k], [self.batch_size] + self.shapes[k])
        return features

    def set_batch(self, batch_num):
        super(TFRecordsDataProvider, self).set_batch(batch_num)


class ParallelBySliceProvider(object):
    def __init__(self,
                 func,
                 kwargs,
                 mode='block',
                 batch_size=256,
                 n_threads=1):        
        self.func = func
        self.kwargs = kwargs
        self.mode = mode
        self.n_threads = n_threads
        self.batch_size = batch_size

    def init_threads(self):
        n = self.n_threads
        ops = []
        tester = self.func(**self.kwargs)
        N = tester.data_length
        testbatch = tester.next()
        labels = tester.labels
        testbatch = zip(labels, testbatch)
        dtypes = {k: v.dtype for k, v in batch}
        shapes = {k: v.shapes[1:] for k, v in batch}
        if self.mode == 'block':
            blocksize = N / n
            ends = [[i * blocksize, (i+1) * blocksize] for i in range(n)]
            ends[-1] = max(ends[-1][1], N)
            subslices = [np.arange(e0, e1) for e0, e1 in ends]
        elif self.mode == 'alternate':
            subslices = [np.arange(N)[i::] for i in range(n)]        
        for n in self.n_threads:
            kwargs = copy.deepcopy(self.kwargs)
            kwargs['subslice'] = subslices[n]
            dp = self.func(**kwargs)
            op = tf.py_func(dp.next, [], [dtypes[k] for k in labels])
            op = {k: op[k]}
            ops.append(op)
        return ops, dtypes, shapes
    
        
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
        self.total_batches = (self.data_length - 1) // self.batch_size + 1
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
              dtypes_dict,
              shapes_dict,
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
        dtypes.append(dtypes_dict[name])
        if shapes_dict is not None:
            shapes.append(shapes_dict[name])
    if shapes_dict is None:
        shapes = None

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
        ops = [{'images': b[0], 'labels': tf.cast(b[1], tf.int32)} for b in batches]
        dtypes = {'images': tf.float32, 'labels': tf.int32}
        shapes = {'images': [784], 'labels': []}
        return ops, dtypes, shapes


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
        super(ImageNet, self).__init__(
            data_path,
            [images, labels],
            batch_size=batch_size,
            postprocess={images: self.postproc_img, labels: self.postproc_labels},
            pad=True,
            *args, **kwargs)
        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

    def postproc_img(self, ims, f):
        norm = ims.astype(np.float32) / 255
        off = np.random.randint(0, 256 - self.crop_size, size=2)
        images_batch = norm[:,
                            off[0]: off[0] + self.crop_size,
                            off[1]: off[1] + self.crop_size]
        return images_batch

    def postproc_labels(self, labels, f):
        return labels.astype(np.int32)

    def next(self):
        batch = super(ImageNet, self).next()
        feed_dict = {'images': np.squeeze(batch[self.group + '/images']),
                     'labels': np.squeeze(batch[self.group + '/labels'])}
        return feed_dict
