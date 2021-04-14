import tensorflow as tf

from tfutils.helper import log
from tfutils.tpu_helper import create_train_estimator_fn, create_train_tpu_config, train_estimator
from tfutils.defaults import DEFAULT_TPU_ZONE, DEFAULT_NUM_SHARDS, DEFAULT_ITERATIONS_PER_LOOP

if tf.__version__ < '1.11':
    # TF 1.9 and below
    from tensorflow.contrib.tpu.python.tpu import tpu_estimator
    tpu_estimator_lib = tpu_estimator
elif tf.__version__ < '2':
    # TF 1.11 and above
    tpu_estimator_lib = tf.contrib.tpu
else:
    # TF 2.0 and above
    tpu_estimator_lib = tf.estimator.tpu

def tpu_train_from_params(params, train_args, use_tpu=False):
    """
    Main tpu training interface function, called by train_from_params in tfutils.train.
    See the doc string there for info.
    """

    # use this for tpu and estimator logging
    tf.logging.set_verbosity(tf.logging.INFO)
    # For convenience, use list of dicts instead of dict of lists
    _params = [{key: value[i] for (key, value) in params.items()}
               for i in range(len(params['model_params']))]
    _trargs = [{key: value[i] for (key, value) in train_args.items()}
               for i in range(len(params['model_params']))]

    param = _params[0]
    trarg = _trargs[0]
    # Support only single model
    assert(len(_params) == 1)
    train_data_params = param['train_params']['data_params']

    model_params = param['model_params']
    lr_params = param['learning_rate_params']
    opt_params = param['optimizer_params']
    loss_params = param['loss_params']
    validation_params = param['validation_params']
    save_params = param['save_params']
    # set up estimator func
    estimator_fn, params_to_pass = create_train_estimator_fn(use_tpu=use_tpu,
                                       model_params=model_params,
                                       lr_params=lr_params,
                                       opt_params=opt_params,
                                       loss_params=loss_params,
                                       validation_params=validation_params)

    if use_tpu:
        if len(param['validation_params'].keys())>0:
            valid_k = list(param['validation_params'].keys())[0]
            validation_data_params = param['validation_params'][valid_k]['data_params']
            eval_batch_size = validation_data_params['batch_size']
        else:
            eval_batch_size = None
        # grab tpu name and gcp, etc from model params
        train_m_config = create_train_tpu_config(model_dir=save_params.get('cache_dir', ''),
                            tpu_name=model_params.get('tpu_name', None),
                            gcp_project=model_params.get('gcp_project', None),
                            steps_per_checkpoint=save_params.get('save_filters_freq', None),
                            tpu_zone=model_params.get('tpu_zone', DEFAULT_TPU_ZONE),
                            num_shards=model_params.get('num_shards', DEFAULT_NUM_SHARDS),
                            keep_checkpoint_max=save_params.get('checkpoint_max', 5),
                            iterations_per_loop=model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP),
                            model_params=model_params)
        train_estimator_classifier = tpu_estimator_lib.TPUEstimator(
                                    use_tpu=True,
                                    model_fn=estimator_fn,
                                    config=train_m_config,
                                    train_batch_size=train_data_params['batch_size'],
                                    eval_batch_size=eval_batch_size,
                                    params=params_to_pass)
        val_estimator_classifier = None

        if model_params.get('num_shards', DEFAULT_NUM_SHARDS) > 8:
            log.info("You are training in pod mode")
            log.info("Setting up validation on a single independent TPU device")
            assert model_params.get('val_tpu_name') is not None
            val_m_config = create_train_tpu_config(model_dir=save_params.get('cache_dir', ''),
                                tpu_name=model_params.get('val_tpu_name', None),
                                gcp_project=model_params.get('gcp_project', None),
                                steps_per_checkpoint=save_params.get('save_filters_freq', None),
                                tpu_zone=model_params.get('val_tpu_zone', DEFAULT_TPU_ZONE),
                                num_shards=8,
                                keep_checkpoint_max=save_params.get('checkpoint_max', 5),
                                iterations_per_loop=model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP),
                                model_params=model_params)

            val_estimator_classifier = tpu_estimator_lib.TPUEstimator(
                                        use_tpu=True,
                                        model_fn=estimator_fn,
                                        config=val_m_config,
                                        train_batch_size=train_data_params['batch_size'],
                                        eval_batch_size=eval_batch_size,
                                        params=params_to_pass)

    else:
        train_estimator_classifier = tf.estimator.Estimator(model_fn=estimator_fn,
                                                            params=params_to_pass)

    return train_estimator(train_cls=train_estimator_classifier,
                           eval_cls=val_estimator_classifier,
                           param=param,
                           trarg=trarg)
