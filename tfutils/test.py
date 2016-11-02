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
    params['model_params'] = {'model_func':  model.alexnet_nonorm}
    params['train_params'] = {'data_func': data.ImageNet,
                              'data_kwargs': {'data_path': DATA_PATH,
                                              'crop_size': 224}}
    params['learning_rate_params'] = {'learning_rate_kwargs': {'learning_rate': 0.01,
                                                               'decay_steps': num_batches_per_epoch,
                                                               'decay_rate': 0.95,
                                                               'staircase': True}}
    params['saver_params'] = {'host': 'localhost',
                              'port': 31001,
                              'dbname': 'tfutils-test',
                              'collname': 'test',
                              'exp_id': 'tfutils-test-3',
                              'save_valid_freq': 20,
                              'save_filters_freq': 100,
                              'cache_filters_freq': 40}
    params['num_steps'] = 460

    return base.run_base(**params)

if __name__ == '__main__':
    main()
