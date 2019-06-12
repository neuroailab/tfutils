import re
import six
import tensorflow as tf
from tfutils.multi_gpu.easy_variable_mgr import COPY_NAME_SCOPE

def get_tf_version_tuple():
    """
    Return TensorFlow version as a 2-element tuple (for comparison).
    """
    return tuple(map(int, tf.__version__.split('.')[:2]))

def crossgpu_batch_norm(inputs,
                    decay=0.9,
                    epsilon=1e-5,
                    beta_initializer=tf.zeros_initializer(),
                    gamma_initializer=tf.ones_initializer(),
                    moving_mean_initializer=tf.zeros_initializer(),
                    moving_variance_initializer=tf.ones_initializer(),
                    data_format='channels_last',
                    add_to_default_updateops=False,
                    updates_collections=None,
                    is_training=True,
                    variables_collections=None,
                    trainable=True,
                    reuse=None,
                    scope=None,
                    verbose=False,
                    gpu_var_string=COPY_NAME_SCOPE,
                    num_dev=None):
                    
    '''
    Cross-GPU BatchNormalization function which computes aggregate E[x] and E[x^2] across gpus to compute the aggregate mean and variance.
    Useful if your batch size is small on any given GPU (cf. https://arxiv.org/pdf/1711.07240.pdf).
    Adapted from: https://github.com/jianlong-yuan/syncbn-tensorflow/blob/master/syncbn.py and https://tensorpack.readthedocs.io/_modules/tensorpack/models/batch_norm.html#BatchNorm
    num_dev is how many gpus you use. Default is None so that user never forgets what to set. Can be set via model_params['num_gpus'].
    NOTE: Assumes same batch size on each GPU.
    '''

    assert num_dev is not None

    TF_version = get_tf_version_tuple()
    assert six.PY2 or TF_version >= (1, 10), \
        "Cross-GPU BatchNorm is only supported in TF>=1.10 ." \
        "Upgrade TF or apply this patch manually: https://github.com/tensorflow/tensorflow/pull/20360"

    if TF_version <= (1, 12):
        try:
            from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
        except Exception:
            pass
        else:
            _validate_and_load_nccl_so()
            
        from tensorflow.contrib.nccl.ops import gen_nccl_ops
    else:
        from tensorflow.python.ops import gen_nccl_ops

    inp_shp = inputs.get_shape().as_list()
    inp_rank = len(inp_shp)
    if inp_rank == 4: # conv layer
        if data_format == 'channels_last':
            red_axises = [0, 1, 2]
            num_outputs = inp_shp[-1]
            fused_data_format = 'NHWC'
        elif data_format == 'channels_first':
            red_axises = [0, 2, 3]
            num_outputs = inp_shp[1]
            fused_data_format = 'NCHW'
        else:
            raise ValueError
    elif inp_rank == 2: # fc layer
        red_axises = [0]
        num_outputs = inp_shp[-1]
    else:
        raise ValueError

    if (updates_collections is None) and (add_to_default_updateops):
        updates_collections = tf.GraphKeys.UPDATE_OPS

    if scope is None:
        scope = 'CrossGPUBatchNorm'

    with tf.variable_scope(
        scope,
        reuse=reuse):

        gamma = tf.get_variable(name='gamma', shape=[num_outputs], dtype=tf.float32,
                                initializer=gamma_initializer, trainable=trainable,
                                collections=variables_collections)

        if verbose:
            gamma = tf.Print(gamma, [gamma], "gamma")

        beta  = tf.get_variable(name='beta', shape=[num_outputs], dtype=tf.float32,
                                initializer=beta_initializer, trainable=trainable,
                                collections=variables_collections)

        if verbose:
            beta = tf.Print(beta, [beta], "beta")

        moving_mean = tf.get_variable(name='moving_mean', shape=[num_outputs], dtype=tf.float32,
                                    initializer=moving_mean_initializer, trainable=False,
                                    collections=variables_collections)
                                    
        moving_var = tf.get_variable(name='moving_variance', shape=[num_outputs], dtype=tf.float32,
                                    initializer=moving_variance_initializer, trainable=False,
                                    collections=variables_collections)

        if is_training and trainable:
          
            if num_dev == 1:
                mean, var = tf.nn.moments(inputs, axes=red_axises)
            else:
                multi_device_gpu_var_string = '/+' + gpu_var_string + '[0-' + str(num_dev-1) + ']'
                shared_name = re.sub(multi_device_gpu_var_string, '', tf.get_variable_scope().name)
                batch_mean        = tf.reduce_mean(inputs, axis=red_axises)

                if verbose:
                    batch_mean = tf.Print(batch_mean, [batch_mean], "input_mean")

                batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axises)

                if verbose:
                    batch_mean_square = tf.Print(batch_mean_square, [batch_mean_square], "input_mean_square")

                batch_mean        = gen_nccl_ops.nccl_all_reduce(
                                        input=batch_mean,
                                        reduction='sum',
                                        num_devices=num_dev,
                                        shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
                batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                                        input=batch_mean_square,
                                        reduction='sum',
                                        num_devices=num_dev,
                                        shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)

                if verbose:
                    batch_mean = tf.Print(batch_mean, [batch_mean], "NCCL_mean")

                mean              = batch_mean
                var               = batch_mean_square - tf.square(batch_mean)

                if verbose:
                    var = tf.Print(var, [var], "NCCL_var")

            outputs = tf.nn.batch_normalization(inputs, 
                                                mean=mean, 
                                                variance=var, 
                                                offset=beta, 
                                                scale=gamma, 
                                                variance_epsilon=epsilon)

            # each gpu will have a copy of the same moving_mean and moving_var variable, which only gets updated once mean and var have been computed across all gpus
            # this way, when tfutils saves the variables (which it only saves the ones on gpu 0) it will save the correct moving_mean and moving_var
            update_moving_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
            update_moving_var_op  = tf.assign(moving_var,  moving_var  * decay + var  * (1 - decay))
        
            if updates_collections is None:
                with tf.control_dependencies([update_moving_mean_op, update_moving_var_op]):
                    outputs = tf.identity(outputs)
            else:
                tf.add_to_collections(updates_collections, update_moving_mean_op)
                tf.add_to_collections(updates_collections, update_moving_var_op)
                outputs = tf.identity(outputs)
        else:
            if inp_rank == 4: # fused batch norm only supports convolutional layer outputs
                outputs,_,_ = tf.nn.fused_batch_norm(inputs, 
                                                     scale=gamma, 
                                                     offset=beta, 
                                                     mean=moving_mean, 
                                                     variance=moving_var, 
                                                     epsilon=epsilon,
                                                     data_format=fused_data_format, 
                                                     is_training=False)
            elif inp_rank == 2:
                outputs = tf.nn.batch_normalization(inputs,
                                                    scale=gamma,
                                                    offset=beta, 
                                                    mean=moving_mean, 
                                                    variance=moving_var,
                                                    variance_epsilon=epsilon)

            else:
                raise ValueError

        return outputs

