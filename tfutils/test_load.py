from __future__ import division, print_function, absolute_import
import os, sys, math, time
from datetime import datetime

import numpy as np
import tensorflow as tf

from tfutils import base, data, model

host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    DATA_PATH = '/om/user/qbilius/imagenet/data.raw'
else:  # agents
    DATA_PATH = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'


def main():
    num_batches_per_epoch = 2**20//256
    params = {}
    params['model_params'] = {'func': model.alexnet_tfutils}
    params['validation_params'] = {'test': {'data': {'func': data.ImageNet,
                                            'data_path': DATA_PATH,
                                            'crop_size': 224},
                                            'num_steps': 10,
                                            'agg_func': np.mean}}
    params['load_params'] = {'host': 'localhost',
                             'port': 31001,
                             'dbname': 'tfutils-test',
                             'collname': 'test',
                             'exp_id': 'tfutils-test-7'}
    params['save_params'] = {'exp_id': 'tfutils-test-7-valid'}

    return base.test_from_params(**params)

if __name__ == '__main__':
    main()
