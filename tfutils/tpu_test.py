import tensorflow as tf

from tfutils.helper import log
from tfutils.tpu_helper import create_test_estimator_fn, create_test_tpu_config, test_estimator
from tfutils.defaults import DEFAULT_TPU_ZONE, DEFAULT_NUM_SHARDS, DEFAULT_ITERATIONS_PER_LOOP
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

def tpu_test_from_params(params, test_args, use_tpu=False):
    """
    Main tpu testing interface function, called by test_from_params in tfutils.test.
    See the doc string there for info.
    """

    # use this for tpu and estimator logging
    tf.logging.set_verbosity(tf.logging.INFO)
    # For convenience, use list of dicts instead of dict of lists
    _params = [{key: value[i] for (key, value) in params.items()}
               for i in range(len(params['model_params']))]
    _ttargs = [{key: value[i] for (key, value) in test_args.items()}
               for i in range(len(params['model_params']))]

    param = _params[0]
    ttarg = _ttargs[0]
    # Support only single model
    assert(len(_params) == 1)

    model_params = param['model_params']
    validation_params = param['validation_params']
    save_params = param['save_params']

    # store a dictionary of estimators, one for each validation params target
    # since may have a different set of eval steps to run on tpu
    # if dict of estimators not feasible, can just create one single estimator
    # and run its predict method multiple times on the same data function in test_estimator (I think)
    cls_dict = {}
    for valid_k in validation_params.keys():
        # set up estimator func
        valid_target_parameter = validation_params[valid_k]
        estimator_fn, params_to_pass = create_test_estimator_fn(use_tpu=use_tpu, 
                                           model_params=model_params,
                                           target_params=valid_target_parameter)
        validation_data_params = valid_target_parameter['data_params']
        eval_val_steps = valid_target_parameter['num_steps']

        if use_tpu:
            # grab tpu name and gcp, etc from model params
            m_config = create_test_tpu_config(model_dir=save_params.get('cache_dir', ''),
                                         eval_steps=eval_val_steps,
                                         tpu_name=model_params.get('tpu_name', None), 
                                         gcp_project=model_params.get('gcp_project', None), 
                                         tpu_zone=model_params.get('tpu_zone', DEFAULT_TPU_ZONE), 
                                         num_shards=model_params.get('num_shards', DEFAULT_NUM_SHARDS),
                                         iterations_per_loop=model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP))

            estimator_classifier = tpu_estimator.TPUEstimator(
                                        use_tpu=True,
                                        model_fn=estimator_fn,
                                        config=m_config,
                                        train_batch_size=validation_data_params['batch_size'],
                                        predict_batch_size=validation_data_params['batch_size'],
                                        params=params_to_pass)

        else:
            estimator_classifier = tf.estimator.Estimator(model_fn=estimator_fn, params=params_to_pass)

        cls_dict[valid_k] = estimator_classifier
    return test_estimator(cls_dict=cls_dict, param=param, ttarg=ttarg)
