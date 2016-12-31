"""
The is the basic illustration of training.
"""
from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime
import pymongo as pm
import numpy as np

import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

from tfutils import base, model, utils

class MNIST(object):
    def __init__(self, batch_size=100, group='train'):
        """
        A specific reader for MNIST stored as a HDF5 file

        Args:
            - data_path: path to imagenet data
            - crop_size: for center crop (crop_size x crop_size)
            - *args: extra arguments for HDF5DataProvider
        Kwargs:
            - **kwargs: extra keyword arguments for HDF5DataProvider
        """
        self.batch_size = batch_size
        data = read_data_sets('/tmp')
        if group == 'train':
            self.data = data.train
        elif group == 'test':
            self.data = data.train
        elif group == 'validation':
            self.data = data.train
        else:
            raise ValueError('MNIST data input "{}" not known'.format(group))

    def __iter__(self):
        return self

    def next(self):
        batch = self.data.next_batch(self.batch_size)
        return {'images': batch[0],
                'labels': batch[1].astype(np.int32)}



num_batches_per_epoch = 10000//256
testhost = 'localhost'
testport = 31001
testdbname = 'tfutils-test'
testcol = 'testcol'

def test_training():
    """This test illustrates how basic training is performed.
       This is the first in a sequence of tests. It creates a database of results that is used
       by the next few tests. 
    """
    #delete old database if it exists
    conn = pm.MongoClient(host=testhost,
                          port=testport)
    conn.drop_database(testdbname)
    nm = testdbname + '_' + testcol + '_training0'
    [conn.drop_database(x) for x in conn.database_names() if x.startswith(nm) and '___RECENT' in x]
    nm = testdbname + '_' + testcol + '_training1'
    [conn.drop_database(x) for x in conn.database_names() if x.startswith(nm) and '___RECENT' in x]

    #set up the parameters
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['save_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0',
                             'save_valid_freq': 20,
                             'save_filters_freq': 200,
                             'cache_filters_freq': 100}
    params['train_params'] = {'data_params': {'func': MNIST,
                                              'batch_size': 100,
                                              'group': 'train'},
                              'queue_params': {'queue_type': 'fifo',
                                               'batch_size': 100,
                                               'n_threads': 4},
                              'num_steps': 500}
    params['learning_rate_params'] = {'learning_rate': 0.05,
                                      'decay_steps': num_batches_per_epoch,
                                      'decay_rate': 0.95,
                                      'staircase': True}
    params['validation_params'] = {'valid0': {'data_params': {'func': MNIST,
                                                              'batch_size': 100,
                                                              'group': 'test'},
                                              'queue_params': {'queue_type': 'fifo',
                                                               'batch_size': 100,
                                                               'n_threads': 4},
                                              'num_steps': 10,
                                              'agg_func': utils.mean_dict}}


    #actually run the training
    base.train_from_params(**params)
    #test if results are as expected
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0'}).count() == 26
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 
            'saved_filters': True}).distinct('step') == [0, 200, 400]

    r = conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 'step': 0})[0]
    asserts_for_record(r, params, train=True)
    r = conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 'step': 20})[0]
    asserts_for_record(r, params, train=True)


    #run another 500 steps
    params['train_params']['num_steps'] = 1000
    base.train_from_params(**params)
    #test if results are as expected
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0'}).count() == 51
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 
         'saved_filters': True}).distinct('step') == [0, 200, 400, 600, 800, 1000]
    assert conn['tfutils-test']['testcol.files'].distinct('exp_id') == ['training0']
    r = conn[testdbname][testcol+'.files'].find({'exp_id': 'training0', 'step': 1000})[0]
    asserts_for_record(r, params, train=True)

    #run 500 more steps but save to a new exp_id
    params['train_params']['num_steps'] = 1500
    params['load_params'] = {'exp_id': 'training0'}
    params['save_params']['exp_id'] = 'training1'
    base.train_from_params(**params)
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'training1', 
                            'saved_filters': True}).distinct('step') == [1200, 1400]


def test_validation():
    """
    This is a test illustrating how to run validation without training.
    This test assumes that test_train has run first (to provide a model to validate).
    """
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['load_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0'}
    params['save_params'] = {'exp_id': 'validation0'}
    params['validation_params'] = {'valid0': {'data_params': {'func': MNIST,
                                                              'batch_size': 100,
                                                              'group': 'test'},
                                              'queue_params': {'queue_type': 'fifo',
                                                               'batch_size': 100,
                                                               'n_threads': 4},
                                              'num_steps': 10,
                                              'agg_func': utils.mean_dict}}

    base.test_from_params(**params)

    conn = pm.MongoClient(host=testhost,
                          port=testport)
    assert conn[testdbname][testcol+'.files'].find({'exp_id': 'validation0'}).count() == 1
    r = conn[testdbname][testcol+'.files'].find({'exp_id': 'validation0'})[0]
    asserts_for_record(r, params, train=False)
    assert r['validates']
    f = r['validation_results']['valid0']['loss']
    idval = conn[testdbname][testcol+'.files'].find({'exp_id': 'training0'})[50]['_id']
    v = conn[testdbname][testcol+'.files'].find({'exp_id': 'validation0'})[0]['validates']
    assert idval == v


def get_extraction_target(inputs, outputs, to_extract, **loss_params):
    """here's how to figure out what names to use:
    names = [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]
    print("NAMES", names)
    """
    targets = {k: tf.get_default_graph().get_tensor_by_name(v) for k, v in to_extract.items()}
    targets['loss'] = utils.get_loss(inputs, outputs, **loss_params)
    return targets


def test_feature_extraction():
    """
    This is a test illustrating how to perform feature extraction.  
    This test assumes that test_train has run first. 
    """
    params = {}
    params['model_params'] = {'func': model.mnist_tfutils}
    params['load_params'] = {'host': testhost,
                             'port': testport,
                             'dbname': testdbname,
                             'collname': testcol,
                             'exp_id': 'training0'}
    params['save_params'] = {'exp_id': 'validation1',
                             'save_intermediate_freq': 1,
                             'save_to_gfs': ['features']}

    targdict = {'func': get_extraction_target,
                'to_extract': {'features': 'validation/valid1/hidden1/fc:0'}}
    targdict.update(base.DEFAULT_LOSS_PARAMS)
    params['validation_params'] = {'valid1': {'data_params': {'func': MNIST,
                                                              'batch_size': 100,
                                                              'group': 'test'},
                                              'queue_params': {'queue_type': 'fifo',
                                                               'batch_size': 100,
                                                               'n_threads': 4},
                                              'targets': targdict,
                                              'num_steps': 10,
                                              'online_agg_func': utils.reduce_mean_dict
                                            }
                                   }
    base.test_from_params(**params)

    conn = pm.MongoClient(host=testhost,
                          port=testport)
    coll = conn[testdbname][testcol+'.files']
    assert coll.find({'exp_id': 'validation1'}).count() == 11
    q = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': True}}
    assert coll.find(q).count() == 1
    r = coll.find(q)[0]
    asserts_for_record(r, params, train=False)
    q1 = {'exp_id': 'validation1', 'validation_results.valid1.intermediate_steps': {'$exists': False}}
    ids = coll.find(q1).distinct('_id')
    assert r['validation_results']['valid1']['intermediate_steps'] == ids


def asserts_for_record(r, params, train=False):
    if r.get('saved_filters'):
        assert r['_saver_write_version'] == 2
        assert r['_saver_num_data_files'] == 1
    assert type(r['duration']) == float

    should_contain = ['save_params', 'load_params', 'model_params', 'validation_params']
    assert set(should_contain).difference(r['params'].keys()) == set()

    vk = r['params']['validation_params'].keys()
    vk1 = r['validation_results'].keys()
    assert set(vk) == set(vk1)

    assert r['params']['model_params']['cfg_initial'] == None
    assert r['params']['model_params']['seed'] == 0
    assert r['params']['model_params']['func']['modname'] == 'tfutils.model'
    assert r['params']['model_params']['func']['objname'] == 'mnist_tfutils'
    assert set(['hidden1', 'hidden2', u'softmax_linear']).difference(r['params']['model_params']['cfg_final'].keys()) == set()

    _k = vk[0]
    should_contain = ['agg_func', 'data_params', 'num_steps', 'online_agg_func', 'queue_params', 'targets']
    assert set(should_contain).difference(r['params']['validation_params'][_k].keys()) == set()
    
    if train:
        assert r['params']['model_params']['train'] == True
        for k in ['num_steps', 'queue_params']:
            assert r['params']['train_params'][k] == params['train_params'][k]

        should_contain = ['loss_params', 'optimizer_params', 'train_params', 'learning_rate_params']
        assert set(should_contain).difference(r['params'].keys()) == set()
        assert r['params']['train_params']['thres_loss'] == 100
        assert r['params']['train_params']['data_params']['func']['modname'] == 'tfutils.tests.test'
        assert r['params']['train_params']['data_params']['func']['objname'] == 'MNIST'

        assert r['params']['loss_params']['agg_func']['modname'] == 'tensorflow.python.ops.math_ops'
        assert r['params']['loss_params']['agg_func']['objname'] == 'reduce_mean'
        assert r['params']['loss_params']['loss_per_case_func']['modname'] == 'tensorflow.python.ops.nn_ops'
        assert r['params']['loss_params']['loss_per_case_func']['objname'] == 'sparse_softmax_cross_entropy_with_logits'
        assert r['params']['loss_params']['targets'] == ['labels']
    else:
        assert not 'train' in r['params']['model_params']
        assert 'train_params' not in r['params']
