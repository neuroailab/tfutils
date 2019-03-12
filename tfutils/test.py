from tfutils.db_interface import DBInterface
from tfutils.helper import parse_params, log
from tfutils.validation import run_all_validations, get_valid_targets_dict
import tensorflow as tf
from tfutils.utils import strip_prefix
from tensorflow.python.ops import variables
import time
from tfutils.defaults import DEFAULT_HOST
from tfutils.tpu_test import tpu_test_from_params

def test(sess,
         dbinterface,
         validation_targets,
         save_intermediate_freq=None):
    """
    Actually runs the testing evaluation loop.

    Args:
        sess (tensorflow.Session): Object in which to run calculations
        dbinterface (DBInterface object): Saver through which to save results
        validation_targets (dict of tensorflow objects): Objects on which validation will be computed.
        save_intermediate_freq (None or int): How frequently to save intermediate results captured during test
            None means no intermediate saving will be saved

    Returns:
        dict: Validation summary.
        dict: Results.

    """
    # Collect args in a dict of lists
    test_args = {
        'dbinterface': dbinterface,
        'validation_targets': validation_targets,
        'save_intermediate_freq': save_intermediate_freq}

    _ttargs = [{key: value[i] for (key, value) in test_args.items()}
               for i in range(len(dbinterface))]

    for ttarg in _ttargs:
        ttarg['dbinterface'].start_time_step = time.time()
        validation_summary = run_all_validations(
                sess,
                ttarg['validation_targets'],
                save_intermediate_freq=ttarg['save_intermediate_freq'],
                dbinterface=ttarg['dbinterface'],
                validation_only=True)

    res = []
    for ttarg in _ttargs:
        ttarg['dbinterface'].sync_with_host()
        res.append(ttarg['dbinterface'].outrecs)

    return validation_summary, res


def test_from_params(load_params,
                     model_params,
                     validation_params,
                     log_device_placement=False,
                     save_params=None,
                     dont_run=False,
                     skip_check=False,
                     use_estimator=False
                     ):
    """
    Main testing interface function.

    Same as train_from_parameters; but just performs testing without training.

    For documentation, see argument descriptions in train_from_params.

    """
    # use tpu only if a tpu_name has been specified
    use_tpu = (model_params.get('tpu_name', None) is not None)
    if use_tpu:
        log.info('Using tpu: %s' %model_params['tpu_name'])

    params, test_args = parse_params(
            'test',
            model_params,
            dont_run=dont_run,
            skip_check=skip_check,
            save_params=save_params,
            load_params=load_params,
            validation_params=validation_params,
            log_device_placement=log_device_placement,
            use_tpu=use_tpu
            )

    # do not need to create sess with estimator interface
    if use_estimator or use_tpu:
        return tpu_test_from_params(params, test_args, use_tpu=use_tpu)
    else:
        with tf.Graph().as_default(), tf.device(DEFAULT_HOST):

            # create session
            sess = tf.Session(
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=log_device_placement,
                        ))

            init_op_global = tf.global_variables_initializer()
            sess.run(init_op_global)
            init_op_local = tf.local_variables_initializer()
            sess.run(init_op_local)
            log.info('Initialized from scratch first')

            # For convenience, use list of dicts instead of dict of lists
            _params = [{key: value[i] for (key, value) in params.items()}
                       for i in range(len(params['model_params']))]
            _ttargs = [{key: value[i] for (key, value) in test_args.items()}
                       for i in range(len(params['model_params']))]


            # Build a graph for each distinct model.
            for param, ttarg in zip(_params, _ttargs):
                print(param['load_params'])
                from_ckpt = param['load_params'].get('from_ckpt')
                use_ckpt = (from_ckpt is not None)

                if not 'cache_dir' in load_params:
                    temp_cache_dir = save_params.get('cache_dir', None)
                    load_params['cache_dir'] = temp_cache_dir
                    log.info('cache_dir not found in load_params, '\
                            + 'using cache_dir ({}) from save_params'.format(
                                temp_cache_dir))

                ttarg['dbinterface'] = DBInterface(
                        var_manager=None,
                        params=param, 
                        load_params=param['load_params'])
                if not use_ckpt:
                    ttarg['dbinterface'].load_rec()
                    ld = ttarg['dbinterface'].load_data
                    assert ld is not None, "No load data found for query, aborting"
                    ld = ld[0]
                    # TODO: have option to reconstitute model_params entirely from
                    # saved object ("revivification")
                    param['model_params']['seed'] = ld['params']['model_params']['seed']
                    cfg_final = ld['params']['model_params']['cfg_final']
                else:
                    cfg_final = param['model_params'].get('cfg_final')

                ttarg['validation_targets'], var_manager \
                        = get_valid_targets_dict(
                            loss_params=None,
                            cfg_final=cfg_final,
                            **param)

                param['load_params']['do_restore'] = True
                param['model_params']['cfg_final'] = cfg_final

                # Build database interface class, loading model 
                ttarg['dbinterface'] = DBInterface(sess=sess,
                                                   params=param,
                                                   var_manager=var_manager,
                                                   load_params=param['load_params'],
                                                   save_params=param['save_params'])
                ttarg['dbinterface'].initialize()

                ttarg['save_intermediate_freq'] \
                        = param['save_params'].get('save_intermediate_freq')

            # Convert back to a dictionary of lists
            params = {key: [param[key] for param in _params]
                      for key in _params[0].keys()}
            test_args = {key: [ttarg[key] for ttarg in _ttargs]
                         for key in _ttargs[0].keys()}

            if dont_run:
                return test_args

            res = test(sess, **test_args)
            sess.close()
            return res
