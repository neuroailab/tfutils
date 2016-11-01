import tensorflow as tf


class ConvNet(object):

    def __init__(self, train=True, seed=0):
        self.train = train
        self.seed = seed
        self.conv_counter = 0
        self.pool_counter = 0
        self.fc_counter = 0
        self.norm_counter = 0
        self.global_counter = 0
        self.parameters = []
        self.architecture = {}

    def conv(self, 
             in_layer,
             out_shape,
             ksize=3,
             stride=1,
             padding='SAME',
             stddev=.01,
             bias=1,
             name=None):
        in_shape = in_layer.get_shape().as_list()[-1]
        self.conv_counter += 1
        self.global_counter += 1
        if name is None:
            name = 'conv' + str(self.conv_counter)
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(
                tf.truncated_normal([ksize, ksize, in_shape, out_shape],
                                     dtype=tf.float32, stddev=stddev,
                                     seed=self.seed),
                name=name+'_weights')
            conv = tf.nn.conv2d(in_layer, 
                                kernel, 
                                [1, stride, stride, 1],
                                padding=padding,
                                name=name)
            biases = tf.Variable(tf.constant(bias,
                                             shape=[out_shape],
                                             dtype=tf.float32), 
                                 name=name+'_biases')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_out = tf.nn.relu(conv_bias, name=scope)
            self.parameters += [kernel, biases]
            self.architecture[name] = {'input': in_layer.name,
                                       'type': 'conv',
                                       'num_filters': out_shape,
                                       'stride': stride,
                                       'kernel_size': ksize,
                                       'padding': padding, 
                                       'bias': bias,
                                       'stddev': stddev,
                                       'seed': self.seed}
            return conv_out

    def norm(self,
             in_layer,
             depth_radius=4,
             bias=1.00, 
             alpha=0.001 / 9.0,
             beta=007,
             name=None):
        self.norm_counter += 1
        self.global_counter += 1
        if name is None:
            name = 'norm' + str(self.norm_counter)
        self.architecture[name] = {'input': in_layer.name,
                                   'type': 'lrnorm',
                                   'depth_radius': depth_radius,
                                   'bias': bias,
                                   'alpha': alpha,
                                   'beta': beta}
        return tf.nn.lrn(in_layer, 
                         depth_radius=depth_radius, 
                         bias=bias, 
                         alpha=alpha, 
                         beta=beta,
                         name=name)

    def fc(self,
           in_layer,
           out_shape,
           dropout=None,
           stddev=.01,
           bias=1,
           name=None):
        in_shape = in_layer.get_shape().as_list()[-1]
        self.fc_counter += 1
        self.global_counter += 1
        if name is None:
            name = 'fc' + str(self.fc_counter)
        # stdevs = [.01,.01,.01] #[.0005, .005, .1]
        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal([in_shape, out_shape],
                                                     dtype=tf.float32,
                                                     stddev=stddev, 
                                                     seed=self.seed),
                                 name=name + '_weights')
            biases = tf.Variable(tf.constant(
                bias,
                shape=[out_shape],
                dtype=tf.float32), name=name + '_biases')
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
            self.architecture[name] = {'input': in_layer.name,
                                       'type': 'fc',
                                       'num_filters': out_shape,
                                       'dropout': dropout,
                                       'bias': bias,
                                       'stddev': stddev,
                                       'seed': self.seed}
            return fc_out

    def pool(self, in_layer, ksize=3, stride=2, padding='VALID', name=None):
        self.pool_counter += 1
        self.global_counter += 1
        if name is None:
            name = 'pool' + str(self.pool_counter)
        self.architecture[name] = {'input': in_layer.name,
                                   'type': 'maxpool',
                                   'kernel_size': ksize,
                                   'stride': stride,
                                   'padding': padding}
        return tf.nn.max_pool(in_layer,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name=name)

    def print_activations(self, t):
        print(t.op.name, ' ', t.get_shape().as_list())


def alexnet(inputs, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(inputs['data'], 64, 11, 4, stddev=.01, bias=0)
    norm1 = m.norm(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool1 = m.pool(norm1, 3, 2)
    conv2 = m.conv(pool1, 64, 192, 5, 1, stddev=.01, bias=1)
    norm2 = m.norm(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool2 = m.pool(norm2, 3, 2)
    conv3 = m.conv(pool2, 192, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 384, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6], name='reshape1')
    m.architecture['reshape1'] = {'inputs': pool5.name}
    fc1 = m.fc(resh1, 256 * 6 * 6, 4096, dropout=.5, stddev=.01, bias=1)
    fc2 = m.fc(fc1, 4096, 4096, dropout=.5, stddev=.01, bias=1)
    fc3 = m.fc(fc2, 4096, 1000, dropout=None, stddev=.01, bias=0)
    return fc3, m.architecture


def alexnet_nonorm(inputs, train=True, cfg_initial=None, seed=0):
    m = ConvNet(train=train, seed=seed)
    conv1 = m.conv(inputs['data'], 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6], name='reshape1')
    m.architecture['reshape1'] = {'inputs': pool5.name}
    fc1 = m.fc(resh1, 4096, dropout=.5, stddev=.01, bias=1)
    fc2 = m.fc(fc1, 4096, dropout=.5, stddev=.01, bias=1)
    fc3 = m.fc(fc2, 1000, dropout=None, stddev=.01, bias=0)
    return fc3, m.architecture


def alexnet_conv(inputs, train=True, seed=0):
    m = ConvNet(train=train, seed=seed)
    conv1 = m.conv(inputs['data'], 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6], name='reshape1')
    m.architecture['reshape1'] = {'inputs': pool5.name}
    return resh1, m.architecture


def alexnet_caffe(inputs, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(inputs['data'], 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 6 * 6], name='reshape1')
    m.architecture['reshape1'] = {'inputs': pool5.name}
    fc1 = m.fc(resh1, 4096, dropout=.5, stddev=.005, bias=1)
    fc2 = m.fc(fc1, 4096, dropout=.5, stddev=.005, bias=1)
    fc3 = m.fc(fc2, 1000, dropout=None, stddev=.01, bias=0)
    return fc3, m.parameters


def alexnet_nocrop(inputs, train=True):
    m = ConvNet(train=train)
    conv1 = m.conv(inputs['data'], 64, 11, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2)
    conv2 = m.conv(pool1, 192, 5, 1, stddev=.01, bias=1)
    pool2 = m.pool(conv2, 3, 2)
    conv3 = m.conv(pool2, 384, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 256, 3, 1, stddev=.01, bias=1)
    conv5 = m.conv(conv4, 256, 3, 1, stddev=.01, bias=1)
    pool5 = m.pool(conv5, 3, 2)
    resh1 = tf.reshape(pool5, [-1, 256 * 7 * 7], name='reshape1')
    m.architecture['reshape1'] = {'inputs': pool5.name}
    fc1 = m.fc(resh1, 4096, dropout=.5, stddev=.01, bias=1)
    fc2 = m.fc(fc1, 4096, dropout=.5, stddev=.01, bias=1)
    fc3 = m.fc(fc2, 1000, dropout=None, stddev=.01, bias=0)
    return fc3, m.architecture


def mnist_tf(inputs, train=True):
    m = ConvNet(train=train, seed=66478)
    conv1 = m.conv(inputs['data'], 64, 7, 4, stddev=.01, bias=0)
    pool1 = m.pool(conv1, 3, 2, padding='VALID')
    conv2 = m.conv(pool1, 256, 5, 1, stddev=.01, bias=0)
    pool2 = m.pool(conv2, 3, 2, padding='VALID')
    conv3 = m.conv(pool2, 512, 3, 1, stddev=.01, bias=0)
    conv4 = m.conv(conv3, 1024, 3, 1, stddev=.01, bias=0)
    conv5 = m.conv(conv4, 512, 3, 1, stddev=.01, bias=0)
    pool5 = m.pool(conv5, 3, 2, padding='VALID')
    resh1 = tf.reshape(pool5, [-1, 512 * 7 * 7], name='reshape1')
    m.architecture['reshape1'] = {'inputs': pool5.name}
    fc1 = m.fc(resh1, 4096, dropout=.5, stddev=.01, bias=.01)
    fc2 = m.fc(fc1, 4096, dropout=.5, stddev=.01, bias=.01)
    fc3 = m.fc(fc2, 1000, dropout=None, stddev=.01, bias=.1)
    return fc3, m.architecture
