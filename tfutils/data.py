from __future__ import absolute_import, division, print_function

import os
import functools
import itertools
import copy
import cPickle
import logging
import threading

import numpy as np
import h5py
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tfutils.utils import isstring

logging.basicConfig()
log = logging.getLogger('tfutils')
log.setLevel('DEBUG')


class DataProviderBase(object):
    """Base class for all data provider objects.   This class must be subclassed to have a
       functional data provider.
    """
    def init_ops(self):
        """
        This method is required of all data providers.   It returns a list of tensorflow
        operations that provide data, and which will be enqueued by the functions in base.py
        """
        raise NotImplementedError()


class ParallelByFileProviderBase(DataProviderBase):
    def __init__(self,
                 source_paths,
                 n_threads=1,
                 shuffle=False,
                 shuffle_seed=0,
                 postprocess=None,
                 read_args=None,
                 read_kwargs=None,
                 trans_dicts=None,
                 **kwargs):
        """
        This is a base class for parallelizing data reading across large groups of (small-ish)
        files.  Different threads read data in different files, which can then be combined into
        a single reading queue (the enqueing of tensorflow ops for each thread is handled
        outside of this class).

        This provider supports the concept of "attribute groups", e.g. different sets of
        data can be put in different files.  For example, "images" could be put in one set
        of files, while "category_labels" could be put in a different set of files.

        To use this class, subclass it and define the get_input_ops method, which will define
        the data-reading method specific to your application.  See docstring of the method below
        for more details.

        Once instantiated, to get operations to enqueue, call the method init_threads (see below).

        Arguments are:
        - source_paths (list of lists of strs): paths of files to be read.  Each element
          of source_paths is a list of files containing batches of data for a group of
          data attributes.  Specifically, source_paths is of the form
             [[path_11, path_12, path_13, ..., path_1n],
              [path_21, path_22, path_23, ..., path_2n],
              ...,
              [path_m1, path_m2, path_m3, ..., path_mn]]

          where m is the number of attribute groups, n is the number of data files, and
          path_ij is the path of the j-th group of batches for the i-th group of data
          attributes.   Parallelization occurs across the data batches, with data from
          corresponding batches in path_1j , path2j, etc... combined together to produce
          a single overall data dictionary for the batch.   It is required that the number of
          paths in each list be the same, and within corresponding paths, that the number of
          batches of data be the same; that is:
                 for all j:
                    num_batches in path_1j = num_batches in path_ij  for all 1 <= i <= m
          Note that for different j, the number of batches can be different.

        - n_threads (int, default=1): number of threads to be used

        - shuffle (bool, default=False): whether to shuffle the order of the paths
          anew on each epoch.

        - shuffle_seed (int, default=0): if shuffle=True, seed to use to initialize the
          random number generator used to generate the random shuffle order

        - read_args (list of tuples or None):  if not None, list of positional arguments to
          pass to data readers, with one such list of arguments for each attribute group

        - read_kwargs (list of dicts or None): if not None, list of keyword arguments to
          pass to data readers, with one such list of arguments for each attribute group

        - postprocess (dict of callables): postprocessing operations to be applied on
          a per-attribute basis.  This is of the form:
             {attr: [(func1, args1, kwargs1), (func2, args2, kwargs2) ...],
              ... }
          The operations func1, func2, etc are applied in the indicated order to the data
          with attribute name attr, with the args and kwargs added:
                  data[attr] = func1(data[attr], *args1, **kwargs1)
                  data[attr] = func2(data[attr], *args2, **kwargs2)
                . ... and so on
          If a given attribute does not appear in this dictionary, no postprocessing is done.

        - trans_dicts (list of dict or None): attribute renaming mappings. If not None,
          this is of the form
               [d_1, d_2, ..., d_m]
          where each d  is either None or a dictionary of the form:
               d_j = {attr_name1: new_attr_name1, attr_name2: new_attr_name2, ... }
          In this case, the keys of d_j must be attribute names in the jth attribute group
          and when produced, data from attr_name1 will be renamed to new_attr_name1 (and so on)
          before being merged into the consolidated data dictionary for each batch.  If a
          given attribute produced by the j-th group does not appear in d_j, the original
          attribute name is retained.

        - **kwargs: any other keyword arguments are simple attached to the object for use
          by subclasses.
        """

        lens = map(len, source_paths)
        assert all([lens[0] == l for l in lens[1:]]), lens
        self.source_paths = source_paths
        self.n_attrs = len(self.source_paths)
        self.n_threads = n_threads
        self.postprocess = {} if postprocess is None else postprocess
        if read_args is not None:
            assert len(read_args) == self.n_attrs
        else:
            read_args = [()] * self.n_atturs
        self.read_args = read_args
        if read_kwargs is not None:
            assert len(read_kwargs) == self.n_attrs
        else:
            read_kwargs = [{} for _ in range(self.n_attrs)]
        self.read_kwargs = read_kwargs
        if trans_dicts is not None:
            assert len(trans_dicts) == len(source_paths)
        self.trans_dicts = trans_dicts
        for _k in kwargs:
            setattr(self, _k, kwargs[_k])

        if self.n_attrs == 1:
            fq = tf.train.string_input_producer(self.source_paths[0],
                                                shuffle=shuffle,
                                                seed=shuffle_seed)
            self.file_queues = [[fq]] * self.n_threads
        else:
            self.file_queues = []
            tuples = zip(*self.source_paths)
            if shuffle:
                rng = np.random.RandomState(seed=shuffle_seed)
                tuples = random_cycle(tuples, rng)
            else:
                tuples = itertools.cycle(tuples)
            tuples = threadsafe_iter(tuples)
            for n in range(self.n_threads):
                coord = Coordinator(tuples, n)
                fqs = []
                for j in range(self.n_attrs):
                    item = Item(coord, j)
                    func = tf.py_func(item.next, [], [tf.string])
                    fq = tf.train.string_input_producer(func, shuffle=False)
                    fqs.append(fq)
                self.file_queues.append(fqs)

    def init_ops(self):
        self.input_ops = []
        for thread_num in range(self.n_threads):
            op = {}
            for attr_num in range(self.n_attrs):
                fq = self.file_queues[thread_num][attr_num]
                args = self.read_args[attr_num]
                kwargs = self.read_kwargs[attr_num]
                _op = self.get_input_op(fq, *args, **kwargs)
                if self.trans_dicts and self.trans_dicts[attr_num]:
                    td = self.trans_dicts[attr_num]
                    for k in td:
                        if k in _op:
                            _op[td[k]] = _op.pop(k)
                op.update(_op)
            self.input_ops.append(op)
        self.apply_postprocessing()
        return self.input_ops

    def get_input_op(self, fq, *args, **kwargs):
        """
        This is the main method that returns a tensorflow data reading operation.

        This method will get called n_threads * n_attrs times in the method init_ops (see above).
        Specifically, it is called once for each thread id and each attribute group.

        The arguments are:
             fq:  filename queue object.  When run in a tf session, this object will act
                  as a queue of filenames.  When fq.dequeue() is called in a tf.Session, it
                  will produce the next filename to begin reading from.   Note: it only makes
                  sense to dequeue from fq if the current file being read has been completed.
             *args: any position arguments to the reader.  these are specified on a
                  per-attribute-group basis (eg. across thread ids, calls for the same attribute
                  group will get the same args).
             *kwargs: any keyward arguments to the reader.  like for *args, these are specified
                  on a per-attribute-group basis.

        As an example of this method, see the TFRecordParallelByFileProvider.get_input_ops.
        """
        raise NotImplementedError()

    def apply_postprocessing(self):
        ops = self.input_ops
        for i in range(len(ops)):
            for source in self.postprocess:
                op = ops[i][source]
                for func, args, kwargs in self.postprocess[source]:
                    op = func(op, *args, **kwargs)
                ops[i][source] = op


DEFAULT_TFRECORDS_GLOB_PATTERN = '*.tfrecords'


def get_data_paths(paths, file_pattern=DEFAULT_TFRECORDS_GLOB_PATTERN):
    if not isinstance(paths, list):
        assert isstring(paths)
        paths = [paths]
    if not isinstance(file_pattern, list):
        assert isstring(file_pattern)
        file_patterns = [file_pattern] * len(paths)
    else:
        file_patterns = file_pattern
    assert len(file_patterns) == len(paths), (file_patterns, paths)
    datasources = []
    for path, file_pattern in zip(paths, file_patterns):
        if os.path.isdir(path):
            tfrecord_pattern = os.path.join(path, file_pattern)
            datasource = tf.gfile.Glob(tfrecord_pattern)
            datasource.sort()
            datasources.append(datasource)
        else:
            datasources.append([path])
    dl = map(len, datasources)
    assert all([dl[0] == d for d in dl[1:]]), dl
    return datasources


def get_parser(shape, dtype):
    dtype = dtype if dtype in [tf.float32, tf.int64] else tf.string
    shape = shape if dtype in [tf.float32, tf.int64] else []
    return tf.FixedLenFeature(shape, dtype)


def parse_standard_tfmeta(paths):
    meta_list = []
    for path in paths:
        if isstring(path):
            if path.startswith('meta') and path.endswith('.pkl'):
                mpaths = [path]
            else:
                assert os.path.isdir(path)
                mpaths = filter(lambda x: x.startswith('meta') and x.endswith('.pkl'),
                                os.listdir(path))
                mpaths = [os.path.join(path, mp) for mp in mpaths]
        else:
            # in this case, it's a list
            assert isinstance(path, list)
            mpaths = path
        d = {}
        for mpath in mpaths:
            d.update(cPickle.load(open(mpath)))
        meta_list.append(d)
    return meta_list


def complete_metadata(meta_dicts, parsed_meta_dicts):
    if meta_dicts is None:
        # if no meta_dicts is passed, just use the saved one for all source_paths
        meta_dicts = parsed_meta_dicts
        log.info('Using all metadata from saved source')
    else:
        assert len(meta_dicts) == len(parsed_meta_dicts)
        for (i, (md, pmd)) in enumerate(zip(meta_dicts, parsed_meta_dicts)):
            # if no meta is passed for this particular source_path, used the saved one
            if md is None:
                meta_dicts[i] = pmd
                log.info('Using saved meta %s for attribute group %d' % (str(pmd), i))
            else:
                if isstring(md):
                    md = {md: None}
                    meta_dicts[i] = md
                elif isinstance(md, list):
                    md = {_m: None for _m in md}
                    meta_dicts[i] = md
                assert hasattr(md, 'keys')
                for k in md:
                    if md[k] is None:
                        md[k] = {}
                    # for all saved metadata keys, if that key is not present in the provide metadata,
                    # add the key from the saved metadata
                    if k in pmd:
                        for _k in pmd[k]:
                            if _k not in md[k]:
                                log.info('Using saved meta for key %s attribute %s for attribute group %d' % (
                                    str(k), str(_k), i))
                                md[k][_k] = pmd[k][_k]

    bad_mds = [i for i, _md in enumerate(meta_dicts) if len(_md) == 0]
    assert len(bad_mds) == 0, 'No metadata specifed for attribute groups: %s' % str(bad_mds)
    return meta_dicts


def merge_meta(meta_dicts, trans_dicts):
    meta_dict = {}
    parser_list = []
    for ind, md in enumerate(meta_dicts):
        parsers = {k: get_parser(md[k]['shape'], md[k]['dtype']) for k in md}
        parser_list.append(parsers)
        if trans_dicts and trans_dicts[ind]:
            td = trans_dicts[ind]
            for k in md:
                if k in td:
                    md[td[k]] = md.pop(k)
        meta_dict.update(md)
    return meta_dict, parser_list


def add_standard_postprocessing(postprocess, meta_dict):
    if postprocess is None:
        postprocess = {}
    for k in meta_dict:
        if k not in postprocess:
            postprocess[k] = []
        dtype = meta_dict[k]['dtype']
        if dtype not in [tf.string, tf.int64, tf.float32]:
            postprocess[k].insert(0, (tf.decode_raw, (meta_dict[k]['dtype'], ), {}))
            postprocess[k].insert(1, (tf.reshape, ([-1] + meta_dict[k]['shape'], ), {}))
    return postprocess


class TFRecordsParallelByFileProvider(ParallelByFileProviderBase):
    def __init__(self,
                 source_dirs,
                 batch_size=256,
                 meta_dicts=None,
                 postprocess=None,
                 trans_dicts=None,
                 file_pattern=DEFAULT_TFRECORDS_GLOB_PATTERN,
                 **kwargs):
        """
        Subclass of ParallelByFileProviderBase specific to TFRecords files.

        The argument source_dirs is a list of directories where files for each attribute
        group reside (one element of source_dir for each attribute group).  The tfrecords files
        in each directory will be detected and passed on to the ParallelByFileProviderBase.

        For each attribute in a tfrecords file, metadata about datatype and shape is needed.
        This class will automatically add postprocessing operations that convert type from raw
        tfrecords strings to the indicated dataypes, and reshape arrays to the indicated shapes.

        This metadata can be supplied two ways.  First, this class supports a "standard form"
        for reading metadata from on-disk pickle files.  Specifically, for each source_dir,
        any files of the form
              source_dir/meta[*].pkl
        will be loaded as python pickle objects and assumed to be dictionaries of metadata
        for the attributes in the files in source_dir. For example, your directory structure
        could look like:
            images_and_labels/
               meta.pkl
                   containing {"images": {"dtype": tf.uint8, "shape": (256, 256, 3)}.
                               "labels": {"dtype": tf.int64, "shape": ()}}
               file1.tfrecords  (containing records with "images" and "labels" keys)
               file2.tfrecords
                ....
            normals/
               meta.pkl
                    containing {"normals": {"dtype": tf.uint8, "shape": (256, 256, 3)}}
               file1.tfrecords (containing records with "normals" key)
               file2.tfrecords
                ....

        If no such metadata pickle files exist, metadata can be supplied by passing the meta_dicts
        argument.

        Arguments:
            - source_dirs (string or list of strings): List of directory names in which
              files reside.  All files inside each directory of the form '*.tfrecords'
              will be read, and the lists of files are sorted before passing to the
              ParallelByFileProviderBase class constructor.   For example:
                  source_dirs = ['/path/to/my/images_and_labels/'
                                 '/path/to/my/normals']
            - batch_size (int, default=256): max size of batches to read (note that read_up_to
              operation might produce fewer than batch_size records).
            - meta_dicts (list of dictionaries, lists strings or None):
              If not None, a list of the same length as source_dirs.  Elements of meta_dicts
              can be dictionaries, lists of strings, or strings.  If dictionaries, then this
              containing metadata for the attributes in the datafiles in the corresponding
              source_dir.  Example:
                  meta_dicts = [{'images': {'dtype' ..., },
                                 'labels': {'dtype' ... },}
                                {'normals': {'dtype': ....}}]
              If an element of meta_dicts is a list of strings, those are treated as the attributes
              to be read from that attribute group (and others ignored).   If the element is a string
              just that attribute is loaded.   For example:
                  meta_dicts = ['images', ['ids', 'means'], ["segmentations"]]
              indicates that the attribute "images" is loaded from the files in the first source_path,
              the attributes "ids" and "means" are loaded for the second source_path, and the attribute
              "segmentations" for the third.
            - file_pattern (str, optional): pattern for selecting files in glob format.

        """
        self.source_dirs = source_dirs
        self.batch_size = batch_size
        parsed_meta_dicts = parse_standard_tfmeta(self.source_dirs)
        self.meta_dicts = complete_metadata(meta_dicts, parsed_meta_dicts)
        self.meta_dict, self.parser_list = merge_meta(self.meta_dicts,
                                                      trans_dicts)
        postprocess = add_standard_postprocessing(postprocess, self.meta_dict)
        source_paths = get_data_paths(source_dirs, file_pattern)
        super(TFRecordsParallelByFileProvider, self).__init__(source_paths,
                                                              read_args=[(p, ) for p in self.parser_list],
                                                              postprocess=postprocess,
                                                              trans_dicts=trans_dicts,
                                                              **kwargs)

    def get_input_op(self, fq, parsers):
        reader = tf.TFRecordReader()
        _, serialized_data = reader.read_up_to(fq, self.batch_size)
        return tf.parse_example(serialized_data, parsers)


class ParallelBySliceProvider(DataProviderBase):
    """
    Data provider for handling parallelization by records within one large randomly-accessible file.
    See an example of usage in tfutils/tests/test_data_hdf5.py.
    """
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

    def init_ops(self):
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


class HDF5DataReader(object):
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
              min_after_dequeue=None,
              seed=0):
    """ A generic queue for reading data
        Built on top of https://indico.io/blog/tensorflow-data-input-part2-extensions/
    """
    if capacity is None:
        capacity = 2 * batch_size
    if min_after_dequeue is None:
        min_after_dequeue = capacity // 2

    names = []
    dtypes = []
    shapes = []

    for name in nodes.keys():
        names.append(name)
        dtypes.append(nodes[name].dtype)
        shapes.append(nodes[name].get_shape()[1:])

    if queue_type == 'random':
        queue = tf.RandomShuffleQueue(capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
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

    def init_ops(self):
        func = functools.partial(self.data.next_batch, self.batch_size)
        batches = [tf.py_func(func, [], [tf.float32, tf.uint8]) for _ in range(self.n_threads)]
        for b in batches:
            b[0].set_shape([self.batch_size, 784])
            b[1].set_shape([self.batch_size])
        ops = [{'images': b[0], 'labels': tf.cast(b[1], tf.int32)} for b in batches]
        return ops


class ImageNet(HDF5DataReader):

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

        if self.group == 'train':
            off = np.random.randint(0, 256 - self.crop_size, size=2)
        else:
            off = int((256 - self.crop_size) / 2)
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


class ImageNetTF(TFRecordsParallelByFileProvider):

    def __init__(self, source_dirs, crop_size=224, **kwargs):
        """
        ImageNet data provider for TFRecords
        """
        self.crop_size = crop_size
        postprocess = {'images': [(self.postprocess_images, (), {})]}
        super(ImageNetTF, self).__init__(source_dirs, postprocess=postprocess, **kwargs)

    def postprocess_images(self, ims):
        def _postprocess_images(im):
            im = tf.decode_raw(im, np.uint8)
            im = tf.image.convert_image_dtype(im, dtype=tf.float32)
            im = tf.reshape(im, [256, 256, 3])
            im = tf.random_crop(im, [self.crop_size, self.crop_size, 3])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)


class Coordinator(object):
    def __init__(self, itr, tid):
        self.curval = {}
        self.itr = itr
        self.tid = tid
        self.lock = threading.Lock()

    def next(self, j):
        with self.lock:
            if not self.curval:
                curval = self.itr.next()
                if not hasattr(curval, 'keys'):
                    n = len(curval)
                    curval = {i: curval[i] for i in range(n)}
                self.curval = curval
            val = self.curval.pop(j)
        return val


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
    local_ls = ls[:]
    while True:
        rng.shuffle(local_ls)
        for e in local_ls:
            yield e


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()
