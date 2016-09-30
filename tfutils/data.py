from __future__ import absolute_import, division, print_function

import tensorflow as tf

import hdf5provider


class ImageNet(hdf5provider.HDF5DataProvider):

    def __init__(self, data_path, subslice, crop_size=None,
                 *args, **kwargs):
        """
        A specific reader for IamgeNet stored as a HDF5 file

        Args:
            - data_path: path to imagenet data
            - subslice: np array for training or eval slice
            - crop_size: for center crop (crop_size x crop_size)
            - *args: extra arguments for HDF5DataProvider
        Kwargs:
            - **kwargs: extra keyword arguments for HDF5DataProvider
        """
        super(ImageNet, self).__init__(
            data_path,
            ['data', 'labels'],
            batch_size=1,  # filll up the queue one image at a time
            subslice=subslice,
            preprocess={'labels': hdf5provider.get_unique_labels},
            postprocess={'data': self.postproc},
            pad=True,
            *args, **kwargs)
        self.data_dict = {'data': tf.placeholder(tf.float32,
                                            shape=(crop_size, crop_size, 3),
                                            name='data'),
                          'labels': tf.placeholder(tf.int64,
                                                   shape=[],
                                                   name='labels')}

    def postproc(self, ims, f):
        norm = ims / 255. - .5
        resh = norm.reshape((3, 256, 256))
        sw = resh.swapaxes(1, 2).swapaxes(2, 3)
        cropped_shape = [self.crop_size, self.crop_size, 3]
        images_batch = tf.random_crop(sw, size=cropped_shape, name='random_crop')
        return images_batch

    def next(self):
        batch = super(ImageNet, self).next()
        feed_dict = {self.data_batch['data']: batch['data'].astype(tf.float32),
                     self.data_batch['labels']: batch['labels'].astype(tf.int64)}
        return feed_dict