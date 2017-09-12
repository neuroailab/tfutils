"""tfutils utils tests.

These tests demonstrate and verify the behaviour of various utilities in tfutils.utils.

"""
import sys
import time
import threading
import Queue

import tensorflow as tf

sys.path.insert(0, "/home/aandonia/tfutils")
import tfutils.error as error
import tfutils.utils as utils


def save_from_queue(queue):
    """Mimic saving a checkpoint from a queue of checkpoints.

    Args:
        queue (Queue.Queue): A queue of checkpoints.

    Raises:
        error.ThreadError: Queue is empty.

    """
    time.sleep(0.1)
    print('Getting checkpoint...')

    try:
        checkpoint = queue.get(block=False)
    except Queue.Empty:
        print('Checkpoint queue is empty...')
        raise error.ThreadError
    else:
        print('Checkpoint {} saved successfully!'.format(checkpoint))


def test_coordinated_thread():
    """Demonstate the use of CoordinatedThread with tf.train.Coordinator.

    A queue is filled with a single 'checkpoint' to be saved by two separate
    threads called sequentially. The first thread will successfully save
    the one and only checkpoint, but the second thread will raise an
    error.ThreadError. This test only passes if that exception is
    reported (can be caught) by the main thread.

    """
    # Start coordinators
    coord = tf.train.Coordinator()
    error_coord = tf.train.Coordinator()

    # Create a queue with only one checkpoint
    queue = Queue.Queue()
    queue.put('Checkpoint_1.ckpt')

    # Create threads
    coord_thread = utils.CoordinatedThread(coord=coord,
                                           target=save_from_queue,
                                           args=(queue,))
    error_thread = utils.CoordinatedThread(coord=error_coord,
                                           target=save_from_queue,
                                           args=(queue,))
    coord_thread.daemon = True
    error_thread.daemon = True

    try:
        coord_thread.start()
        coord.join([coord_thread])
    finally:
        assert queue.empty()
        assert not coord_thread.is_alive()

    try:
        error_thread.start()
        error_coord.join([error_thread])
    except error.ThreadError:
        print('A ThreadError was successfully raised.')
    else:
        raise('A ThreadError should have been raised.')
    finally:
        assert not error_thread.is_alive()


def test_aggregation():
    num_gpus = 2
    batch_size = 2
    num_elements = 2
    minibatch_size = batch_size / num_gpus

    tensor = tf.ones((minibatch_size, 2, 2))
    list_ = num_elements * [tensor]
    dict_ = {
        'tensor': tensor,
        'list': list_,
        'dict': {
            'tensor': tensor,
            'list': [
                {'tensor': tensor},
                [tensor, tensor],
            ]
        },
    }

    tensors = num_gpus * [tensor]
    dicts = num_gpus * [dict_]
    lists = num_gpus * [list_]

    tensor_output = utils.aggregate_outputs(tensors)
    assert isinstance(tensor_output, tf.Tensor)
    assert tensor_output.shape == (batch_size, 2, 2)

    dict_output = utils.aggregate_outputs(dicts)
    assert isinstance(dict_output, dict)
    assert isinstance(dict_output['tensor'], tf.Tensor)
    assert dict_output['tensor'].shape == (batch_size, 2, 2)
    assert isinstance(dict_output['list'], list)
    assert isinstance(dict_output['list'][0], tf.Tensor)
    assert dict_output['list'][0].shape == (batch_size, 2, 2)
    assert len(dict_output['list']) == num_elements
    assert isinstance(dict_output['dict'], dict)
    assert isinstance(dict_output['dict']['tensor'], tf.Tensor)
    assert dict_output['dict']['tensor'].shape == (batch_size, 2, 2)
    assert isinstance(dict_output['dict']['list'], list)
    assert isinstance(dict_output['dict']['list'][0], dict)
    assert isinstance(dict_output['dict']['list'][0]['tensor'], tf.Tensor)
    assert dict_output['dict']['list'][0]['tensor'].shape == (batch_size, 2, 2)
    assert isinstance(dict_output['dict']['list'][1], list)
    assert isinstance(dict_output['dict']['list'][1][0], tf.Tensor)
    assert dict_output['dict']['list'][0]['tensor'].shape == (batch_size, 2, 2)

    list_output = utils.aggregate_outputs(lists)
    assert(isinstance(list_output, list))
    assert(len(list_output) == num_elements)
    assert(isinstance(list_output[0], tf.Tensor))
    assert(list_output[0].shape == (batch_size, 2, 2))

    class TestClass(object):
        def __init__(self, data):
            self.data = data

    myobj = TestClass(tensor)
    objs = num_gpus * [myobj]

    try:
        utils.aggregate_outputs(objs)
    except TypeError:
        print('A TypeError was successfully raised.')
    else:
        raise('A TypeError should have been raised.')


if __name__ == '__main__':

    test_coordinated_thread()
    test_aggregation()
