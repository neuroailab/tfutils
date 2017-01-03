from __future__ import division, print_function, absolute_import

import numpy as np
from tfutils import data


def test_queues():
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
    assert batch['data'].shape == (data_batch_size, 5,5,3)
    assert queue.nodes['data'].get_shape().as_list() == [data_batch_size] + shape
    assert queue.batch['data'].get_shape().as_list() == [batch_size] + shape


if __name__ == '__main__':
    test_queues()