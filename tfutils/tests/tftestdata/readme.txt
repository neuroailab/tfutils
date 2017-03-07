Test for tf records reader:

Two sets of attribute files have been created:
Set 1 contains images and ids, stored in folder images/:
Keys:   ['images', 'ids']
dtypes: [tf.string, tf.int64] converted from [tf.uint8, tf.int64]
Set 2 contains the mean of the images and ids, stored in folder means/.
Keys:   ['means', 'ids']
dtypes: [tf.float32, tf.int64]
Matching attributes will have matching file names in the respective subfolders.

There are a total of 7 files per attribute, each containing a 
different amount of batches.

Statistics:
n_records = 1600
batch_size = 16
n_batches = 100
batch_split = [20, 3, 32, 10, 12, 5, 18]
image_height = 32
image_width = 32
image_channels = 3
image_type = uint8
mean_type = float
id_type = int

Task: Match the attributes of the correct batches and output them.
A match is correct if the "id" of attribute "images" agrees with 
the "id" of the attribute "means".

The test files have been created with "create_tfrecords_single.py" on "freud".
An example test can be found "test_tfrecords.py".
