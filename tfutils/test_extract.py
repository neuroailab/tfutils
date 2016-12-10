from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime

import numpy as np
import tensorflow as tf

from tfutils import base, data, model, utils

host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    DATA_PATH = '/om/user/qbilius/imagenet/data.raw'
else:  # agents
    DATA_PATH = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'


def get_targets(inputs, outputs, **params):
    #names = [[x.name for x in op.values()] for op in tf.get_default_graph().get_operations()]
    #print("NAMES", names)
    f = tf.get_default_graph().get_tensor_by_name('validation/test/conv1/Conv2D:0')
    targets = {'loss': utils.get_loss(inputs, outputs, **params),
               'feature_mean': tf.reduce_mean(f)}
    return targets

targdict = {'func': get_targets}
targdict.update(base.default_loss_params())

def main():
    num_batches_per_epoch = 2**20//256
    params = {}
    params['model_params'] = {'func': model.alexnet_tfutils}
    params['validation_params'] = {'test': {'data': {'func': data.ImageNet,
                                                     'data_path': DATA_PATH,
                                                     'crop_size': 224},
                                            'num_steps': 10,
                                            'targets': targdict
                                        }
                               }
    params['load_params'] = {'host': 'localhost',
                             'port': 31001,
                             'dbname': 'tfutils-test',
                             'collname': 'test',
                             'exp_id': 'tfutils-test-7'}
    params['save_params'] = {'exp_id': 'tfutils-test-7-feats'}

    return base.test_from_params(**params)

if __name__ == '__main__':
    main()
