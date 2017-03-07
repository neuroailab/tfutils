import h5py
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import json

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

N = 1600 #2048000
BATCH_SIZE = 16.0
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
NEW_WIDTH = 32
NEW_HEIGHT = 32
CHANNELS = 3

data_dir = '/media/data/one_world_dataset'
data2_dir = '/media/data2/one_world_dataset'

data_file = os.path.join(data_dir, 'dataset.hdf5')
f = h5py.File(data_file, 'r')

output_dir = os.path.join(data2_dir, "tftestdata/images")

def resize_img(image):
    return np.array(Image.fromarray(image).resize((NEW_HEIGHT, NEW_WIDTH), Image.BICUBIC))

def mean_img(image):
    return np.mean(np.array(image).astype(np.float))

def parse_action(action):
    action = json.loads(action)
    # parsed action vector
    pact = []
    # pact[0] : teleport random
    if 'teleport_random' in action and action['teleport_random'] is True:
	pact.append(1)
    else:
	pact.append(0)
    # pact[1:4] : agent velocity
    if 'vel' in action:
	pact.extend(action['vel'])
    else:
	pact.extend(np.zeros(3))
    # pact[4:7] : agent angular velocity
    if 'ang_vel' in action:
	pact.extend(action['ang_vel'])
    else:
	pact.extend(np.zeros(3))
    # pact[7:25] : actions
    if 'actions' in action:
	# fill object actions vector
	object_actions = []
	for objact in action['actions']:
	    if 'force' in objact:
		object_actions.extend(objact['force'])
	    else:
		object_actions.extend(np.zeros(3))
	    if 'torque' in objact:
		object_actions.extend(objact['torque'])
	    else:
		object_actions.extend(np.zeros(3))
	    """
		The chosen object not necessarily the one acted upon
		depending on action_pos. The actual object acted upon
		is stored in 'id'
	    """
	    if 'object' in objact:
		object_actions.append(int(objact['object']))
	    else:
		object_actions.append(0)
	    if 'action_pos' in objact:
		act_pos = objact['action_pos']
		act_pos[0] = int(act_pos[0] / float(SCREEN_HEIGHT) * NEW_HEIGHT)
		act_pos[1] = int(act_pos[1] / float(SCREEN_WIDTH) * NEW_WIDTH)
		object_actions.extend(act_pos)
	    else:
		object_actions.extend(np.zeros(2))
	""" 
		Each object action vector has a length of 3+3+1+2=9.
		Object actions are performed on maximally 2 objects
		simultanously (CRASHING action). Thus, the vector length
		has to be 2*9=18
	"""
	while len(object_actions) < 18:
	    object_actions.append(0)
	# append object actions vector
	pact.extend(object_actions)
    return np.array(pact)

k = 0
batches = [20, 3, 32, 10, 12, 5, 18]
output_file = os.path.join(output_dir, str(k) + '.tfrecords')
writer = tf.python_io.TFRecordWriter(output_file)
k += 1
for t in range(N):
    b = t / BATCH_SIZE
    if b >= sum(batches[:k]):
        writer.close()
	output_file = os.path.join(output_dir, str(k) + '.tfrecords')
	writer = tf.python_io.TFRecordWriter(output_file)
        print("%f" % (float(t) / N * 100))
        k += 1
        if k > len(batches):
            break;

    # tfrecords datum
    img = resize_img(f['images'][t])
    mea = mean_img(img)
    #hgt = img.shape[0]
    #wdt = img.shape[1]
    #cha = img.shape[2]
    img = img.tostring()
    #nor = resize_img(f['normals'][t]).tostring()
    #obj = resize_img(f['objects'][t]).tostring()
    #inf = f['worldinfo'][t]
    #act = f['actions'][t]
    #par = parse_action(act).tostring()
    #val = f['valid'][t]
    idx = t
    datum = tf.train.Example(features=tf.train.Features(feature={
           'images': _bytes_feature(img),
#           'normals': _bytes_feature(nor),
#           'objects': _bytes_feature(obj),
#           'height': _int64_feature(hgt),
#           'width': _int64_feature(wdt),
#           'channels': _int64_feature(cha),
#           'worldinfo': _bytes_feature(inf),
#           'actions': _bytes_feature(act),
#	   'parsed_actions': _bytes_feature(par),
           'ids': _int64_feature(int(idx)),
#           'means': _float_feature(float(mea)),
	}))
    writer.write(datum.SerializeToString())
writer.close()
f.close()
