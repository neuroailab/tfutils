from __future__ import division, print_function, absolute_import

import os, tempfile
from collections import Counter

import numpy as np
import h5py
import tensorflow as tf

from tfutils import data


def create_hdf5(total_size):
    tempf = tempfile.NamedTemporaryFile(suffix='.hdf5', dir='/tmp', delete=False)
    tempf.close()
    with h5py.File(tempf.name, 'w') as f:
        f.create_dataset('data', (total_size, ), dtype=np.int64)
        f['data'][:] = np.arange(total_size)
    return tempf.name


class DataHDF5(data.HDF5DataProvider):

    def __init__(self, path=None, batch_size=5, total_size=100):
        self.kind = 'hdf5 read (feed_dict)'
        self.batch_size = batch_size
        self.data_path = create_hdf5(total_size=total_size)

        super(DataHDF5, self).__init__(self.data_path,
                                       ['data'],
                                       batch_size=self.batch_size,
                                       pad=True)

    def next(self):
        batch = super(DataHDF5, self).next()
        feed_dict = {'data': np.squeeze(batch['data'])}
        return feed_dict

    def cleanup(self):
        os.remove(self.data_path)


def test_queue_unique():
    batch_size = 100
    data_batch_size = 10
    total_size = batch_size * 10

    # 2*total_size so that we don't go back to the start of the data
    data_iter = DataHDF5(batch_size=data_batch_size, total_size=2*total_size)

    queue = data.Queue(data_iter, batch_size=batch_size)
    with h5py.File(data_iter.data_path) as f:
        data_from_file = f['data'][:]

    sess = tf.Session()
    queue.start_threads(sess)
    data_from_queue = []
    for i in range(total_size // batch_size):
        batch = queue.batch['data'].eval(session=sess)
        data_from_queue.extend(batch.tolist())

    queue.stop_threads(sess)
    sess.close()
    data_iter.cleanup()

    # Are values in order?
    # assert np.all(data_from_file[:total_size] == data_from_queue)
    # Are values unique?
    assert np.all([v == 1 for k,v in Counter(data_from_queue).items()])


def test_queue_size():
    """
    This test checks if queues are producing batches of correct size.
    """
    batch_size = 256
    shape = [5, 5, 3]

    # batch size of 1 (special case)
    data_batch_size = 1
    data_iter = [{'data': np.random.random(shape)} for i in range(1)]
    queue = data.Queue(data_iter, data_batch_size=data_batch_size, batch_size=batch_size)
    batch = queue.next()
    assert list(batch['data'].shape) == shape
    assert queue.nodes['data'].get_shape().as_list() == shape
    assert queue.batch['data'].get_shape().as_list() == [batch_size] + shape

    # batch size of 32 (general case)
    data_batch_size = 32
    data_iter = [{'data': np.random.random((data_batch_size, 5, 5, 3))} for i in range(1)]
    queue = data.Queue(data_iter, data_batch_size=data_batch_size, batch_size=batch_size)
    batch = queue.next()
    assert batch['data'].shape == (data_batch_size, 5, 5, 3)
    assert queue.nodes['data'].get_shape().as_list() == [data_batch_size] + shape
    assert queue.batch['data'].get_shape().as_list() == [batch_size] + shape


if __name__ == '__main__':
    test_queue_unique()
    test_queue_size()
