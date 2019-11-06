"""
This script contains class method `dataset_func` which will return dataset elements

The data format structure of ImageNet required for `dataset_func` is similar as
data structure generated by following structure:
    https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py

The only difference is that each tfrecords file only contains two attributes:
    images: jpeg format of images
    labels: int64 of 0-999 labels
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import sys
import numpy as np

def fetch_dataset(filename):
    """
    Useful util function for fetching records
    """
    buffer_size = 32 * 1024 * 1024  # 32 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def color_normalize(image):
    image = tf.cast(image, tf.float32) / 255
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std
    return image


class ImageNet(object):
    """
    Class where data provider for ImageNet will be built
    """

    TRAIN_LEN = 1281167
    VAL_LEN = 50000

    def __init__(
        self,
        image_dir,
        prep_type,
        crop_size=224,
        smallest_side=256,
        resize=None,
        is_train=True,
        drop_remainder=False
    ):
        self.image_dir = image_dir

        # Parameters about preprocessing
        self.prep_type = prep_type
        self.crop_size = crop_size
        self.smallest_side = smallest_side
        self.resize = resize
        self.num_cores = 8

        self.drop_remainder = drop_remainder
        self.is_train = is_train

        # Placeholders to be filled later
        self.on_tpu = None
        self.file_pattern = None

    def get_tfr_filenames(self):
        """
        Get list of tfrecord filenames
        for given folder_name fitting the given file_pattern
        """
        assert self.file_pattern, "Please specify file pattern!"
        tfrecord_pattern = os.path.join(self.image_dir, self.file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()
        return np.asarray(datasource)

    def get_resize_scale(self, height, width):
        """
        Get the resize scale so that the shortest side is `smallest_side`
        """
        smallest_side = tf.convert_to_tensor(
            self.smallest_side, dtype=tf.int32)

        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)

        scale = tf.cond(
            tf.greater(height, width),
            lambda: smallest_side / width,
            lambda: smallest_side / height,
        )
        return scale

    def resize_cast_to_uint8(self, image):
        image = tf.cast(
            tf.image.resize_bilinear(
                [image], [self.crop_size, self.crop_size])[0],
            dtype=tf.uint8,
        )
        image.set_shape([self.crop_size, self.crop_size, 3])
        return image

    def central_crop_from_jpg(self, image_string):
        """
        Resize the image to make its smallest side to be 256;
        then get the central 224 crop
        """
        shape = tf.image.extract_jpeg_shape(image_string)
        scale = self.get_resize_scale(shape[0], shape[1])
        cp_height = tf.cast(self.crop_size / scale, tf.int32)
        cp_width = tf.cast(self.crop_size / scale, tf.int32)
        cp_begin_x = tf.cast((shape[0] - cp_height) / 2, tf.int32)
        cp_begin_y = tf.cast((shape[1] - cp_width) / 2, tf.int32)
        bbox = tf.stack([cp_begin_x, cp_begin_y, cp_height, cp_width])
        crop_image = tf.image.decode_and_crop_jpeg(
            image_string, bbox, channels=3)
        image = self.resize_cast_to_uint8(crop_image)

        return image

    def resnet_crop_from_jpg(self, image_str):
        """
        Random crop in Inception style, see GoogLeNet paper, also used by ResNet
        """
        shape = tf.image.extract_jpeg_shape(image_str)
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32, shape=[1, 1, 4])
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
            area_range=(0.08, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True,
        )

        # Get the cropped image
        bbox_begin, bbox_size, bbox = sample_distorted_bounding_box
        random_image = tf.image.decode_and_crop_jpeg(
            image_str,
            tf.stack([bbox_begin[0], bbox_begin[1],
                      bbox_size[0], bbox_size[1]]),
            channels=3,
        )
        bad = _at_least_x_are_equal(shape, tf.shape(random_image), 3)

        # central crop if bad
        min_size = tf.minimum(shape[0], shape[1])
        offset_height = tf.random_uniform(
            shape=[], minval=0, maxval=shape[0] - min_size + 1, dtype=tf.int32
        )
        offset_width = tf.random_uniform(
            shape=[], minval=0, maxval=shape[1] - min_size + 1, dtype=tf.int32
        )
        bad_image = tf.image.decode_and_crop_jpeg(
            image_str,
            tf.stack([offset_height, offset_width, min_size, min_size]),
            channels=3,
        )
        image = tf.cond(bad, lambda: bad_image, lambda: random_image)

        image = self.resize_cast_to_uint8(image)
        return image

    def alexnet_crop_from_jpg(self, image_string):
        """
        Resize the image to make its smallest side to be 256;
        then randomly get a 224 crop
        """
        shape = tf.image.extract_jpeg_shape(image_string)
        scale = self.get_resize_scale(shape[0], shape[1])
        cp_height = tf.cast(self.crop_size / scale, tf.int32)
        cp_width = tf.cast(self.crop_size / scale, tf.int32)

        # Randomly sample begin x and y
        x_range = [0, shape[0] - cp_height + 1]
        y_range = [0, shape[1] - cp_width + 1]
        if self.prep_type == "alex_center":
            # Original AlexNet preprocessing uses center 256*256 to crop
            min_shape = tf.minimum(shape[0], shape[1])
            x_range = [
                tf.cast((shape[0] - min_shape) / 2, tf.int32),
                shape[0]
                - cp_height
                + 1
                - tf.cast((shape[0] - min_shape) / 2, tf.int32),
            ]
            y_range = [
                tf.cast((shape[1] - min_shape) / 2, tf.int32),
                shape[1] - cp_width + 1 -
                tf.cast((shape[1] - min_shape) / 2, tf.int32),
            ]

        cp_begin_x = tf.random_uniform(
            shape=[], minval=x_range[0], maxval=x_range[1], dtype=tf.int32
        )
        cp_begin_y = tf.random_uniform(
            shape=[], minval=y_range[0], maxval=y_range[1], dtype=tf.int32
        )

        bbox = tf.stack([cp_begin_x, cp_begin_y, cp_height, cp_width])
        crop_image = tf.image.decode_and_crop_jpeg(
            image_string, bbox, channels=3)
        image = self.resize_cast_to_uint8(crop_image)

        return image

    def preprocessing(self, image_string):
        """
        Preprocessing for each image
        """
        assert self.is_train is not None, "Must specify is_train"

        def _rand_crop(image_string):
            if self.prep_type == "resnet":
                image = self.resnet_crop_from_jpg(image_string)
            else:
                image = self.alexnet_crop_from_jpg(image_string)

            return image

        if self.prep_type == "inception":
            if self.resize is None:
                inception_image_size = 299
            else:
                inception_image_size = self.resize

            import tfutils.inception_preprocessing as inception_preprocessing
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = inception_preprocessing.preprocess_image(image,
                                                             is_training=self.is_train,
                                                             image_size=inception_image_size)
        else:
            if self.is_train:
                image = _rand_crop(image_string)
                image = tf.image.random_flip_left_right(image)
            else:
                image = self.central_crop_from_jpg(image_string)

            image = color_normalize(image)

            if self.resize is not None:
                image = tf.image.resize_images(image, [self.resize, self.resize], align_corners=True)
        return image

    def data_parser(self, value):
        """
        Parse record and preprocessing
        """
        # Load the image and preprocess it
        keys_to_features = {
            "images": tf.FixedLenFeature((), tf.string, ""),
            "labels": tf.FixedLenFeature([], tf.int64, -1),
        }
        parsed = tf.parse_single_example(value, keys_to_features)
        image_string = parsed["images"]
        image_label = parsed["labels"]

        # Do the preprocessing
        image = self.preprocessing(image_string)
        if self.on_tpu:
            return image, image_label
        else:
            return {"images": image, "labels": image_label}

    def process_dataset(self, dataset):
        # if training, shuffle. repeat indefinitely
        if self.is_train:
            dataset = dataset.shuffle(self.q_cap)
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat()

        # re-shuffle if training
        if self.is_train:
            # Read each file
            dataset = dataset.apply(
                tf.contrib.data.parallel_interleave(
                    fetch_dataset, cycle_length=self.num_cores, sloppy=True
                )
            )

            dataset.shuffle(self.q_cap)
        else:
            # Read each file, but make it deterministic for validation
            dataset = dataset.apply(
                tf.contrib.data.parallel_interleave(
                    fetch_dataset, cycle_length=self.num_cores, sloppy=False
                )
            )

        # apply preprocessing to each image
        dataset = dataset.map(
            self.data_parser,
            num_parallel_calls=64)

        dataset = dataset.prefetch(4)
        if self.drop_remainder:
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        else:
            dataset = dataset.batch(self.batch_size)

        return dataset

    def dataset_func(
        self,
        is_train,
        batch_size,
        q_cap=51200,
        file_pattern="train-*"
    ):
        """
        Build the dataset, get the elements
        """
        self.on_tpu = False
        self.is_train = is_train
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.q_cap = q_cap

        tfr_list = self.get_tfr_filenames()
        if self.is_train:
            dataset = tf.data.Dataset.list_files(tfr_list)
        else:
            dataset = tf.data.Dataset.list_files(tfr_list, shuffle=False)

        dataset = self.process_dataset(dataset)

        # Batch the dataset and make iteratior
        next_element = dataset.make_one_shot_iterator().get_next()
        return next_element

    def dataset_func_tpu(
            self, params):
        """
        Build the dataset, get the elements
        """
        self.on_tpu = True
        self.drop_remainder = True

        if self.is_train:
            self.file_pattern = 'train-*'
        else:
            self.file_pattern = 'validation-*'
        self.batch_size = params['batch_size']
        self.q_cap = 1024

        # get file pattern and create dataset
        file_pattern = os.path.join(
            self.image_dir, self.file_pattern)

        if self.is_train:
            dataset = tf.data.Dataset.list_files(file_pattern)
        else:
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

        dataset = self.process_dataset(dataset)

        dataset = dataset.prefetch(2)
        images, labels = dataset.make_one_shot_iterator().get_next()
        return images, labels

