import tensorflow as tf


class ConvNet(object):

    def __init__(self, train=True, seed=None):
        self.train = train
        self.seed = seed
        self.conv_counter = 0
        self.pool_counter = 0
        self.fc_counter = 0
        self.global_counter = 0
        self.parameters = []

    def conv(self, in_layer, in_shape, out_shape, ksize=3, stride=1,
             padding='SAME', stddev=.01, bias=1):
        self.conv_counter += 1
        self.global_counter += 1
        name = 'conv' + str(self.conv_counter)
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(
                tf.truncated_normal([ksize, ksize, in_shape, out_shape],
                                     dtype=tf.float32, stddev=stddev,
                                     seed=self.seed),
                name='weights')

            conv = tf.nn.conv2d(in_layer, kernel, [1, stride, stride, 1],
                                padding=padding)
            biases = tf.Variable(tf.constant(
                bias,
                shape=[out_shape],
                dtype=tf.float32), name='biases')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_out = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [kernel, biases]
            return conv_out

    def norm(self, in_layer):
        return tf.nn.lrn(in_layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    def fc(self, in_layer, in_shape, out_shape, dropout=None, stddev=.01, bias=1):
        self.fc_counter += 1
        self.global_counter += 1
        name = 'fc' + str(self.fc_counter)
        # stdevs = [.01,.01,.01] #[.0005, .005, .1]
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([in_shape, out_shape],
                                                     dtype=tf.float32,
                                              stddev=stddev, seed=self.seed),
                                 name='weights')
            biases = tf.Variable(tf.constant(
                bias,
                shape=[out_shape],
                dtype=tf.float32), name='biases')
            if dropout is None:
                fcm = tf.matmul(in_layer, kernel)
                fc_out = tf.nn.bias_add(fcm, biases, name=scope)
            elif not self.train:
                fc_out = tf.nn.relu_layer(in_layer, kernel, biases, name=scope)
            else:
                fcr = tf.nn.relu_layer(in_layer, kernel, biases)
                # fck = tf.matmul(in_layer, kernel)
                # bias = tf.nn.bias_add(fck, biases)
                # fcr = tf.nn.relu(bias)
                fc_out = tf.nn.dropout(fcr, dropout, seed=self.seed, name=scope)
            self.parameters += [kernel, biases]
            return fc_out

    def pool(self, in_layer, ksize=3, stride=2, padding='VALID'):
        self.pool_counter += 1
        name = 'pool' + str(self.pool_counter)
        return tf.nn.max_pool(in_layer,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name=name)

    def print_activations(self, t):
        print(t.op.name, ' ', t.get_shape().as_list())


def alexnet(images, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(images, 3, 64, 11, 4, stddev=.01, bias=0)
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    pool1 = m.pool(norm1, 3, 2)
    conv2 = m.conv(pool1, 64, 192, 5, 1, stddev=.01, bias=1)
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    pool2 = m.pool(norm2, 3, 2)
    conv3 = m.conv(pool2, 192, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 384, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    fc1 = m.fc(resh1, 256 * 6 * 6, 4096, dropout=.5, stddev=.01, bias=1)
    fc2 = m.fc(fc1, 4096, 4096, dropout=.5, stddev=.01, bias=1)
    fc3 = m.fc(fc2, 4096, 1000, dropout=None, stddev=.01, bias=0)

    return fc3, m.parameters


def alexnet_nonorm(images, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(images, 3, 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 64, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 192, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 384, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    fc1 = m.fc(resh1, 256 * 6 * 6, 4096, dropout=.5, stddev=.01, bias=1)
    fc2 = m.fc(fc1, 4096, 4096, dropout=.5, stddev=.01, bias=1)
    fc3 = m.fc(fc2, 4096, 1000, dropout=None, stddev=.01, bias=0)

    return fc3, m.parameters


def alexnet_conv(images, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(images, 3, 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 64, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 192, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 384, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])

    return resh1, m.parameters


def alexnet_caffe(images, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(images, 3, 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 64, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 192, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 384, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6])
    fc1 = m.fc(resh1, 256 * 6 * 6, 4096, dropout=.5, stddev=.005, bias=1)
    fc2 = m.fc(fc1, 4096, 4096, dropout=.5, stddev=.005, bias=1)
    fc3 = m.fc(fc2, 4096, 1000, dropout=None, stddev=.01, bias=0)

    return fc3, m.parameters


def alexnet_nocrop(images, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(images, 3, 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 64, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 192, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 384, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 7 * 7])
    fc1 = m.fc(resh1, 256 * 7 * 7, 4096, dropout=.5, stddev=.01, bias=1)
    fc2 = m.fc(fc1, 4096, 4096, dropout=.5, stddev=.01, bias=1)
    fc3 = m.fc(fc2, 4096, 1000, dropout=None, stddev=.01, bias=0)

    return fc3, m.parameters


def mnist_tf(images, train=True):
    m = ConvNet(train=train, seed=66478)
    conv1 = m.conv(images, 3, 64, 7, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2, padding='VALID')
    conv2 = m.conv(pool1, 64, 256, 5, 1, stddev=.01, bias=0)
    pool2 = m.pool(conv2, 3, 2, padding='VALID')
    conv3 = m.conv(pool2, 256, 512, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 512, 1024, 3, 1, stddev=.01, bias=0)
    conv5 = m.conv(conv4, 1024, 512, 3, 1, stddev=.01, bias=0)
    pool5 = m.pool(conv5, 3, 2, padding='VALID')
    resh1 = tf.reshape(pool5, [-1, 512 * 7 * 7])
    fc1 = m.fc(resh1, 512 * 7 * 7, 4096, dropout=.5, stddev=.01, bias=.01)
    fc2 = m.fc(fc1, 4096, 4096, dropout=.5, stddev=.01, bias=.01)
    fc3 = m.fc(fc2, 4096, 1000, dropout=None, stddev=.01, bias=.1)

    return fc3, m.parameters