import tensorflow as tf
import numpy as np
from PIL import Image
import os

base_path = '/media/data2/one_world_dataset/tftestdata'
tfimg_ptr = tf.python_io.tf_record_iterator(path=os.path.join(base_path, 'images/0.tfrecords'))
tfm_ptr = tf.python_io.tf_record_iterator(path=os.path.join(base_path, 'means/0.tfrecords'))

datum = tf.train.Example()
test_passed = True
for i in range(20*16):
    #parse images file
    im = tfimg_ptr.next()
    datum.ParseFromString(im)
    img_string = (datum.features.feature['images'].bytes_list.value[0])
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    img = img_1d.reshape((32, 32, -1))
    #Image.fromarray(img).save('test.png')
    img_mean = np.mean(np.array(img).astype(np.float))
    id_img = (datum.features.feature['ids'].int64_list.value[0])
    
    #parse means file
    m = tfm_ptr.next()
    datum.ParseFromString(m)
    
    m_mean = (datum.features.feature['means'].float_list.value[0])
    id_m = (datum.features.feature['ids'].int64_list.value[0])

    if(id_m != id_img):
        print('id test failed on %d' % i)
        print('image id %d' % id_img)
        print('mean id %d' % id_m)
        test_passed = False
        break

    if((m_mean - img_mean) > 0.01):
        print('mean test failed on %d' % i)
        print('loaded mean: %f' % img_mean)
        print('computed mean: %f' % m_mean)
        test_passed = False
        break

if(test_passed):
    print('TEST PASSED!')
else:
    print('TEST FAILED!')
