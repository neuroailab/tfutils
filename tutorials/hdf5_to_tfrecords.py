import h5py
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import json
import sys
import os
import cPickle

'''
This script takes an HDF5 file as an input and creates multiple 
tfrecords files out of it.

Each attribute will be split into a separate folder.
Within each attribute folder there will be multiple 
tfrecords files, each containing 4 batches of batch size 256 of data.

Besides, one pickeled meta data file "meta.pkl", containing 
the key, shape and data type of the attribute will be created in
each attribute folder.

Additionally each image will be resized to 256x256, which is not necessary
but useful to do before writing the tfrecords file

args:
    - input hdf5 file
    - output directory
'''

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_shapes_and_dtypes(data):
    shapes = {}
    dtypes = {}
    for k in data.keys():
        if isinstance(data[k][0], str):
            shapes[k] = []
            dtypes[k] = tf.string
        elif isinstance(data[k][0], np.ndarray):
            shapes[k] = data[k][0].shape
            dtypes[k] = tf.uint8
        elif isinstance(data[k][0], np.bool_):
            shapes[k] = []
            dtypes[k] = tf.string
        else:
            raise TypeError('Unknown data type', type(data[k][0]))
    return shapes, dtypes

def resize_img(image, shape):
    return np.array(Image.fromarray(image).resize(shape, Image.BICUBIC))

if __name__ == '__main__':
    batch_size = 256
    batches_per_file = 4
    new_shape = [256, 256, 3]

    input_file = sys.argv[1] #e.g. '/media/data/one_world_dataset/dataset8.hdf5'
    assert input_file.endswith('.hdf5')

    output_dir = sys.argv[2] #e.g. '/media/data2/one_world_dataset/tfvaldata'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    f = h5py.File(input_file, 'r')
    output_keys = f.keys()
    assert len(output_keys) > 0, 'At least one output key is needed: ' + str(output_keys)

    dlen = [len(f[k]) for k in f.keys()]
    assert len(np.unique(dlen)) == 1, 'All HDF5 entries do need to have\
                                    the same length: ' + str(dlen)
    N = dlen[0]
    assert N > 0, 'At least one entry needed per key: ' + str(N)

    # extract shapes and data types from hdf5 file
    shapes, dtypes = get_shapes_and_dtypes(f)

    # since we will be resizing all images to new_shapes later on 
    # we have to adjust the size
    for key  in shapes:
        if len(shapes[key]) == 3 and shapes[key][2] == 3:
            shapes[key] = new_shape 

    # create meta.pkl files
    print('Creating meta data pickle files...')
    for output_key in output_keys:
        output_path = os.path.join(output_dir, output_key)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_file = os.path.join(output_path, 'meta.pkl')
        meta_dict = {}
        meta_dict[output_key] = {'dtype': dtypes[output_key], 
                                 'shape': shapes[output_key]}
        with open(output_file, 'w') as meta_file:
            cPickle.dump(meta_dict, meta_file)

    num_batches = int(float(N) / batch_size / batches_per_file)
    batches = [4] * num_batches
    print('Writing %d entries into %d batches...' % (N, sum(batches)))

    # initialize writers 
    k = 0
    output_files = []
    for output_key in output_keys:
        output_files.append(os.path.join(output_dir, output_key, str(k) + '.tfrecords'))
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    # write files
    k += 1
    for t in range(N):
        b = t / batch_size
        if b >= sum(batches[:k]):
            for writer in writers:
                writer.close()
            for i in range(len(output_files)):
	        output_files[i] = os.path.join(output_dir, \
                                          output_keys[i], str(k) + '.tfrecords')
            for i in range(len(writers)):
	        writers[i] = tf.python_io.TFRecordWriter(output_files[i])
            print("%f %%" % (float(t) / N * 100))
            k += 1
            if k > len(batches):
                break;

        for i in range(len(output_keys)):
            data = f[output_keys[i]][t]
            if isinstance(data, np.ndarray):
                if len(data.shape) == 3 and \
                        data.shape[2] == 3:
                    data = resize_img(data, new_shape[0:2])
                data = data.tostring()
            elif isinstance(data, np.bool_):
                data = str(1 if data else 0)
            datum = tf.train.Example(features=tf.train.Features(feature={
                    output_keys[i]: _bytes_feature(data),
            }))
            writers[i].write(datum.SerializeToString())

    '''
    Side note: Data can be not only stored as _bytes_feature but also as
        - _int64_feature
        - _float_feature
    e.g.
        datum = tf.train.Example(features=tf.train.Features(feature={
                'your_key1': _int64_feature(int(your_int))
                'your_key2': _float_feature(float(your_float))
                }))
    '''
    
    for writer in writers:
        writer.close()
    f.close()
    print("100 %")
