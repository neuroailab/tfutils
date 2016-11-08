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
    params['train_params'] = {'data': {'func': data.ImageNet,
                                       'data_path': DATA_PATH,
                                       'crop_size': 224}}
    params['learning_rate_params'] = {'learning_rate': 0.01,
                                      'decay_steps': num_batches_per_epoch,
                                      'decay_rate': 0.95,
                                      'staircase': True}
    params['saver_params'] = {'host': 'localhost',
                              'port': 31001,
                              'dbname': 'tfutils-test',
                              'collname': 'test',
                              'exp_id': 'tfutils-test-6',
                              'save_valid_freq': 20,
                              'save_filters_freq': 100,
                              'cache_filters_freq': 80}
    params['num_steps'] = 230

    return base.run_base(**params)

if __name__ == '__main__':
    main()
