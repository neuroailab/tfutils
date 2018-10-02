from __future__ import absolute_import, division, print_function
import inspect
from functools import wraps
from collections import OrderedDict
from contextlib import contextmanager

import tensorflow as tf

def initializer(kind='xavier', *args, **kwargs):
    if kind == 'xavier':
        init = tf.contrib.layers.xavier_initializer(*args, **kwargs)
    else:
        init = getattr(tf, kind + '_initializer')(*args, **kwargs)
    return init

def batchnorm_corr(inputs, is_training, decay = 0.999, epsilon = 1e-3):

    scale = tf.get_variable(name = 'scale', shape = [inputs.get_shape()[-1]], initializer = tf.ones_initializer(), trainable=True)
    beta = tf.get_variable(name = 'beta', shape = [inputs.get_shape()[-1]], initializer = tf.zeros_initializer(), trainable=True)
    pop_mean = tf.get_variable(name = 'bn_mean', shape = [inputs.get_shape()[-1]], initializer = tf.zeros_initializer(), trainable=False)
    pop_var = tf.get_variable(name = 'bn_var', shape = [inputs.get_shape()[-1]], initializer = tf.ones_initializer(), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, list(range(inputs.get_shape().ndims - 1)))
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            output = tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        output = tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

    return output

def conv(inp,
         out_depth,
         ksize=[3,3],
         strides=[1,1,1,1],
         data_format='channels_last',
         padding='SAME',
         kernel_init='xavier',
         kernel_init_kwargs=None,
         bias=0,
         weight_decay=None,
         activation='relu',
         batch_norm=False,
         is_training=False,
         init_zero=None,
         dropout=None,
         dropout_seed=0,
         name='conv'
         ):

    # BATCH_NORM_DECAY = 0.9
    # BATCH_NORM_EPSILON = 1e-5
    
    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')
    init = initializer(kind='constant', value=bias)
    biases = tf.get_variable(initializer=init,
                            shape=[out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='bias')
    # ops
    if dropout is not None:
        inp = tf.nn.dropout(inp, dropout, seed=dropout_seed, name='dropout')

    conv = tf.nn.conv2d(inp, kernel,
                        strides=strides,
                        padding=padding)
    output = tf.nn.bias_add(conv, biases, name=name)

    if batch_norm:
        # # if activation is none, should use zeros; else ones
        # if init_zero is None:
        #     init_zero = True if activation is None else False
        # if init_zero: 
        #     gamma_init = tf.zeros_initializer()
        # else:
        #     gamma_init = tf.ones_initializer()

        # axis = 1 if data_format == 'channels_first' else 3
        # output = tf.layers.batch_normalization(inputs=output,
        #                                        axis=axis,
        #                                        momentum=BATCH_NORM_DECAY,
        #                                        epsilon=BATCH_NORM_EPSILON,
        #                                        center=True,
        #                                        scale=True,
        #                                        training=is_training,
        #                                        trainable=True,
        #                                        fused=True,
        #                                        gamma_initializer=gamma_init,
        #                                        name="post_conv_BN")

        output = batchnorm_corr(output, is_training=is_training, decay = 0.999, epsilon = 1e-3)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)

    return output

def conv_bnf(inp,
         out_depth,
         ksize=[3,3],
         strides=[1,1,1,1],
         padding='SAME',
         kernel_init='xavier',
         kernel_init_kwargs=None,
         bias=0,
         weight_decay=None,
         activation='relu6',
         batch_norm=True,
         is_training=True,
         name='conv_bnf'
         ):

    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]
        
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')

    # ops
    conv = tf.nn.conv2d(inp, kernel,
                        strides=strides,
                        padding=padding)

    if batch_norm:
        output = batchnorm_corr(output, is_training=is_training, decay = 0.999, epsilon = 1e-3)
    else:
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')
        output = tf.nn.bias_add(conv, biases, name=name)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)

    return output

def depthsep_conv(inp,
             out_depth,
             multiplier=1,
             ksize=3,
             strides=1,
             dep_padding='SAME',
             sep_padding='SAME',
             batch_norm = True,
             is_training=True,
             name='depthsep_conv',
             *args,
             **kwargs
             ):

    with tf.variable_scope('depthwise_conv'):
        d_out = depth_conv(inp, multiplier = multiplier,
                ksize = ksize,
                strides = strides,
                padding = dep_padding,
                batch_norm = batch_norm,
                is_training = is_training,
                *args, **kwargs)

    with tf.variable_scope('pointwise_conv'):
        # we batch norm first according to mobilenet paper
        p_out = conv_bnf(d_out, out_depth = out_depth,
                ksize = 1,
                strides = 1,
                padding = sep_padding,
                batch_norm = batch_norm,
                is_training = is_training,
                *args, **kwargs)
        
    return p_out

def depth_conv(inp,
             multiplier=1,
             ksize=3,
             strides=1,
             padding='SAME',
             kernel_init='xavier',
             kernel_init_kwargs=None,
             bias=1,
             activation='relu6',
             weight_decay=None,
             batch_norm = True,
             is_training=True,
             name='depth_conv'
             ):

    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(strides, int):
        strides = [1, strides, strides, 1]

    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    
    in_depth = inp.get_shape().as_list()[-1]

    out_depth = multiplier * in_depth

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], in_depth, multiplier],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')

    conv = tf.nn.depthwise_conv2d(inp, kernel,
                            strides=strides,
                            padding=padding)
        
    if batch_norm:
        output = batchnorm_corr(output, is_training=is_training, decay = 0.999, epsilon = 1e-3)
    else:
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')
        output = tf.nn.bias_add(conv, biases, name=name)   

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)

    return output

def fc(inp,
       out_depth,
       kernel_init='xavier',
       kernel_init_kwargs=None,
       bias=1,
       weight_decay=None,
       activation='relu',
       batch_norm=False,
       is_training=False,
       dropout=None,
       dropout_seed=0,
       name='fc'):

    if weight_decay is None:
        weight_decay = 0.
    # assert out_shape is not None
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    resh = tf.reshape(inp, [inp.get_shape().as_list()[0], -1], name='reshape')
    in_depth = resh.get_shape().as_list()[-1]

    # weights
    init = initializer(kernel_init, **kernel_init_kwargs)
    kernel = tf.get_variable(initializer=init,
                            shape=[in_depth, out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights')
    init = initializer(kind='constant', value=bias)
    biases = tf.get_variable(initializer=init,
                            shape=[out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='bias')

    # ops
    if dropout is not None:
        resh = tf.nn.dropout(resh, dropout, seed=dropout_seed, name='dropout')
    fcm = tf.matmul(resh, kernel)
    output = tf.nn.bias_add(fcm, biases, name=name)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)
    if batch_norm:
        output = batchnorm_corr(output, is_training=is_training, decay = 0.999, epsilon = 1e-3)
    return output


def global_pool(inp, kind='avg', keep_dims=False, name=None):
    if kind not in ['max', 'avg']:
        raise ValueError('Only global avg or max pool is allowed, but'
                            'you requested {}.'.format(kind))
    if name is None:
        name = 'global_{}_pool'.format(kind)
    h, w = inp.get_shape().as_list()[1:3]
    out = getattr(tf.nn, kind + '_pool')(inp,
                                    ksize=[1,h,w,1],
                                    strides=[1,1,1,1],
                                    padding='VALID')
    if keep_dims:
        output = tf.identity(out, name=name)
    else:
        output = tf.reshape(out, [out.get_shape().as_list()[0], -1], name=name)
        
    return output

def avg_pool2d(inp, kernel_size, stride=2, padding='VALID', name=None):
    if name is None:
        name = 'avg_pool2d'
    output = tf.contrib.layers.avg_pool2d(inp, kernel_size=kernel_size, stride=stride, padding=padding)
    return output

class ConvNet(object):

    INTERNAL_FUNC = ['arg_scope', '_func_wrapper', '_val2list', 'layer',
                     '_reuse_scope_name', '__call__', '_get_func']
    CUSTOM_FUNC = [conv, fc, global_pool, conv_bnf, depthsep_conv, depth_conv, avg_pool2d]

    def __init__(self, defaults=None, name=None):
        """
        A quick convolutional neural network constructor

        This is wrapper over many tf.nn functions for a quick construction of
        a standard convolutional neural network that uses 2d convolutions, pooling
        and fully-connected layers, and most other tf.nn methods.

        It also stores layers and their parameters easily accessible per
        tfutils' approach of saving everything.

        Kwargs:
            - defaults
                Default kwargs values for functions. Complimentary to `arg_scope
            - name (default: '')
                If '', then the existing scope is used.
        """
        self._defaults = defaults if defaults is not None else {}
        self.name = name
        self.state = None
        self.output = None
        self._layer = None
        self.layers = OrderedDict()
        self.params = OrderedDict()
        self._scope_initialized = False

    def __getattribute__(self, attr):
        attrs = object.__getattribute__(self, '__dict__')
        internal_func = object.__getattribute__(self, 'INTERNAL_FUNC')

        if attr in attrs:  # is it an attribute?
            return attrs[attr]
        elif attr in internal_func:  # is it one of the internal functions?
            return object.__getattribute__(self, attr)
        else:
            func = self._get_func(attr)
            return self._func_wrapper(func)

    def _get_func(self, attr):
        custom_func = object.__getattribute__(self, 'CUSTOM_FUNC')
        custom_func_names = [f.__name__ for f in custom_func]
        if attr in custom_func_names:  # is it one of the custom functions?
            func = custom_func[custom_func_names.index(attr)]
        else:
            func = getattr(tf.nn, attr)  # ok, so it is a tf.nn function
        return func

    def _func_wrapper(self, func):
        """
        A wrapper on top of *any* function that is called.

        - Pops `inp` and `layer` from kwargs,
        - All args are turned into kwargs
        - Default values from arg_scope are set
        - Sets the name in kwargs to func.__name__ if not specified
        - Expands `strides` from an int or list inputs for
          all functions and expands `ksize` for pool functions.

        If `layer` is not None, a new scope is created, else the existing scope
        is reused.

        Finally, all params are stored.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs['func_name'] = func.__name__

            # convert args to kwargs
            varnames = inspect.getargspec(func).args
            for i, arg in enumerate(args):
                kwargs[varnames[i+1]] = arg  # skip the first (inputs)

            layer = kwargs.pop('layer', self._layer)
            if layer not in self.params:
                self.params[layer] = OrderedDict()

            # update kwargs with default values defined by user
            if func.__name__ in self._defaults:
                kwargs.update(self._defaults[func.__name__])

            if 'name' not in kwargs:
                fname = func.__name__
                if fname in self.params[layer]:
                    if fname in self.params[layer]:
                        i = 1
                        while fname + '_{}'.format(i) in self.params[layer]:
                            i += 1
                        fname += '_{}'.format(i)
                kwargs['name'] = fname

            spec = ['avg_pool', 'max_pool', 'max_pool_with_argmax']
            if 'ksize' in kwargs and func.__name__ in spec:
                kwargs['ksize'] = self._val2list(kwargs['ksize'])
            if 'strides' in kwargs:
                kwargs['strides'] = self._val2list(kwargs['strides'])

            self.params[layer][kwargs['name']] = kwargs

        return wrapper

    def __call__(self, inp=None):
        output = inp
        for layer, params in self.params.items():
            with tf.variable_scope(layer):
                for func_name, kwargs in params.items():
                    with tf.variable_scope(func_name):
                        output = kwargs.get('inp', output)
                        if output is None:
                            raise ValueError('Layer {} function {} got None as input'.format(layer, func_name))
                        kw = {k:v for k,v in kwargs.items() if k not in ['func_name', 'inp']}
                        func = self._get_func(kwargs['func_name'])
                        output = tf.identity(func(output, **kw), name='output')
                self.layers[layer] = tf.identity(output, name='output')
        self.output = output
        return output


    def _val2list(self, value):
        if isinstance(value, int):
            out = [1, value, value, 1]
        elif len(value) == 2:
            out = [1, value[0], value[1], 1]
        else:
            out = value
        return out

    @contextmanager
    def arg_scope(self, defaults):
        """
        Sets the arg_scope.

        Pass a dict of {<func_name>: {<arg_name>: <arg_value>, ...}, ...}. These
        values will then override the default values for the specified functions
        whenever that function is called.
        """
        self._defaults = defaults
        yield
        self._defaults = {}

    @contextmanager
    def layer(self, name):
        """
        Sets the scope. Can be used with `with`.
        """
        if name is None or name == '':
            raise ValueError('Layer name cannot be None or an empty string')
        self._layer = name
        yield

    def _reuse_scope_name(self, name):
        graph = tf.get_default_graph()
        if graph._name_stack is not None and graph._name_stack != '':
            name = graph._name_stack + '/' + name + '/'  # this will reuse the already-created scope
        else:
            name += '/'
        return name


def mnist(train=True, seed=0):
    m = ConvNet()
    with m.arg_scope({'fc': {'kernel_init': 'truncated_normal',
                             'kernel_init_kwargs': {'stddev': .01, 'seed': seed},
                             'dropout': None, 'batch_norm': False}}):
        m.fc(128, layer='hidden1')
        m.fc(32, layer='hidden2')
        m.fc(10, activation=None, layer='softmax_linear')

    return m


def alexnet(train=True, norm=True, seed=0, **kwargs):
    defaults = {'conv': {'batch_norm': False,
                         'kernel_init': 'xavier',
                         'kernel_init_kwargs': {'seed': seed}},
                         'weight_decay': .0005,
                'max_pool': {'padding': 'SAME'},
                'fc': {'batch_norm': False,
                       'kernel_init': 'truncated_normal',
                       'kernel_init_kwargs': {'stddev': .01, 'seed': seed},
                       'weight_decay': .0005,
                       'dropout_seed': 0}}
    m = ConvNet(defaults=defaults)
    dropout = .5 if train else None

    m.conv(96, 11, 4, padding='VALID', layer='conv1')
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    m.max_pool(3, 2, layer='conv1')

    m.conv(256, 5, 1, layer='conv2')
    if norm:
        m.lrn(depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')
    m.max_pool(3, 2, layer='conv2')

    m.conv(384, 3, 1, layer='conv3')
    m.conv(384, 3, 1, layer='conv4')

    m.conv(256, 3, 1, layer='conv5')
    m.max_pool(3, 2, layer='conv5')

    m.fc(4096, dropout=dropout, bias=.1, layer='fc6')
    m.fc(4096, dropout=dropout, bias=.1, layer='fc7')
    m.fc(1000, activation=None, dropout=None, bias=0, layer='fc8')

    return m


def mnist_tfutils(inputs, train=True, **kwargs):
    m = mnist(train=train)
    return m(inputs['images']), m.params


def alexnet_tfutils(inputs, **kwargs):
    m = alexnet(**kwargs)
    return m(inputs['images']), m.params
