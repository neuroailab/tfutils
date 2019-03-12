import time
import tensorflow as tf
from tfutils.db_interface import DBInterface
from tfutils.helper import log
from tfutils.optimizer import ClipOptimizer
from tfutils.defaults import DEFAULT_TPU_ZONE, DEFAULT_NUM_SHARDS, DEFAULT_ITERATIONS_PER_LOOP, DEFAULT_TPU_LOSS_PARAMS

# tpu and estimator imports
from tensorflow.contrib.tpu.python.tpu import tpu_config, tpu_estimator
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

def train_estimator(cls,
                    param,
                    trarg):

    model_dir = param['save_params'].get('cache_dir', '')
    train_steps = param['train_params']['num_steps']
    # only single targets during eval mode
    need_val = len(param['validation_params'].keys())>0
    steps_per_eval = param['save_params'].get('save_valid_freq')
    if need_val:
        valid_k = param['validation_params'].keys()[0]
        validation_data_params = param['validation_params'][valid_k]['data_params']
        valid_steps = param['validation_params'][valid_k]['num_steps']
        valid_fn = validation_data_params['func']
        if steps_per_eval is None:
            steps_per_eval = param['save_params']['save_filters_freq']
        else:
            save_filters_freq = param['save_params'].get('save_filters_freq')
            if save_filters_freq is not None:
                # these need to be the same right now because estimator loads
                # from last checkpoint after validating
                assert(steps_per_eval == save_filters_freq)
            else:
                param['save_params']['save_filters_freq'] = steps_per_eval
    train_fn = param['train_params']['data_params']['func'] 

    model_params = param['model_params']
    iterations_per_loop = model_params.get('iterations_per_loop', DEFAULT_ITERATIONS_PER_LOOP)

    if (steps_per_eval is None) or (steps_per_eval < iterations_per_loop): # eval steps cannot be less than TPU iterations
        log.info('Setting save_valid_freq ({}) to be the same as iterations_per_loop ({}).'.format(steps_per_eval, iterations_per_loop))
        steps_per_eval = iterations_per_loop

    train_hooks = param['train_params'].get('hooks')
    if need_val:
        valid_hooks = param['validation_params'][valid_k].get('hooks')
    else:
        valid_hooks = None

    current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)
    # initialize db here (currently no support for loading and saving to different places. May need to modify init so load_params can load from different dir, estimator interface limited
    #    when loading and saving to different paths, may need to create a new config)

    trarg['dbinterface'] = DBInterface(sess=None,
                                   params=param,
                                   global_step=current_step,
                                   save_params=param['save_params'],
                                   load_params=param['load_params'],
                                   cache_dir=model_dir)


    log.info('Training beginning ...')
    log.info('Training for %d steps. Current '
                    'step %d' % (train_steps,
                                 current_step))

    trarg['dbinterface'].start_time_step = time.time()

    tpu_validate_first = param['train_params'].get('tpu_validate_first', False)
    def do_tpu_validation():
        log.info('Starting to evaluate.')
        eval_results = cls.evaluate(
          input_fn=valid_fn,
          hooks=valid_hooks,
          steps=valid_steps)
        log.info('Saving eval results to database.')
        trarg['dbinterface'].save(valid_res={valid_k: eval_results}, validation_only=True)
        log.info('Done saving eval results to database.')

        return eval_results

    if tpu_validate_first:
        eval_results = do_tpu_validation()

    while current_step < train_steps:
        next_eval = min(current_step + steps_per_eval,
                            train_steps)

        log.info('Training until step %d' % next_eval)
        cls.train(
        input_fn=train_fn, max_steps=next_eval, hooks=train_hooks)
        current_step = next_eval

        if need_val:
            eval_results = do_tpu_validation()
    
    # sync with hosts
    res = []
    trarg['dbinterface'].sync_with_host()
    res.append(trarg['dbinterface'].outrecs)
    # returning final eval results for convenience
    return eval_results, res

def test_estimator(cls_dict,
                    param,
                    ttarg):

    # load params query stores path to checkpoint
    if param['load_params']['do_restore'] and (param['load_params']['query'] is not None):
        # path to specific checkpoint
        load_dir = param['load_params']['query']
    else:
        # gets latest checkpoint from model_dir
        load_dir = None

    ttarg['dbinterface'] = DBInterface(sess=None,
                                   params=param,
                                   save_params=param['save_params'],
                                   load_params=param['load_params'])


    ttarg['dbinterface'].start_time_step = time.time()

    m_predictions = {}
    for valid_k in cls_dict.keys():
        cls = cls_dict[valid_k]
        validation_data_params = param['validation_params'][valid_k]['data_params']
        # can use to filter particular params to save, if not there will set to None and all saved
        filter_keys = param['validation_params'][valid_k].get('keys_to_save') 
        session_hooks = param['validation_params'][valid_k].get('hooks') 
        valid_fn = validation_data_params['func']
        log.info('Starting to evaluate ({}).'.format(valid_k))
        eval_results = cls.predict(
          input_fn=valid_fn,
          predict_keys=filter_keys,
          hooks=session_hooks,
          checkpoint_path=load_dir)
        m_predictions[valid_k] = list(eval_results)

    log.info('Saving eval results to database.')
    # set validation only to be True to just save the results and not filters
    ttarg['dbinterface'].save(valid_res=m_predictions, validation_only=True)
    log.info('Done saving eval results to database.')
    
    # sync with hosts
    res = []
    ttarg['dbinterface'].sync_with_host()
    res.append(trarg['dbinterface'].outrecs)
    # returning final eval results for convenience
    return eval_results, res

def create_train_estimator_fn(use_tpu, 
                        model_params,
                        lr_params,
                        opt_params,
                        loss_params,
                        validation_params):

    """
    Creates a model_fn for use with tf.Estimator for training and eval.
    """
    # set up loss params 
    loss_agg_func = loss_params.get('agg_func', DEFAULT_TPU_LOSS_PARAMS['agg_func'])

    loss_per_case_func = loss_params.get('loss_per_case_func', DEFAULT_TPU_LOSS_PARAMS['loss_per_case_func'])

    loss_func_kwargs = loss_params.get('loss_func_kwargs', {})

    loss_agg_func_kwargs = loss_params.get('agg_func_kwargs', {})

    # tells clip optimizer to use tpu
    if ((opt_params.get('func', None) is not None) and opt_params['func'].__name__ == 'ClipOptimizer') or ((opt_params.get('optimizer', None) is not None) and opt_params['optimizer'].__name__ == 'ClipOptimizer'):
        opt_params['use_tpu'] = use_tpu

    # build params dictionary to be instantiated with the model_fn
    params_to_pass = {}
    params_to_pass['model_params'] = model_params
    params_to_pass['opt_params'] = opt_params
    params_to_pass['loss_agg_func'] = loss_agg_func
    params_to_pass['loss_per_case_func'] = loss_per_case_func
    params_to_pass['loss_func_kwargs'] = loss_func_kwargs
    params_to_pass['loss_agg_func_kwargs'] = loss_agg_func_kwargs
    params_to_pass['lr_params'] = lr_params

    def model_fn(features, labels, mode, params):
        model_params = params['model_params']
        opt_params = params['opt_params']
        loss_agg_func = params['loss_agg_func']
        loss_per_case_func = params['loss_per_case_func']
        loss_func_kwargs = params['loss_func_kwargs']
        loss_agg_func_kwargs = params['loss_agg_func_kwargs']
        lr_params = params['lr_params']

        model_params['train'] = (mode==tf.estimator.ModeKeys.TRAIN)
        if opt_params['use_tpu']:
            model_params['batch_size'] = params['batch_size'] # per shard batch_size

        model_func = model_params.pop('func')

        outputs = model_func(inputs=features, **model_params)
        if isinstance(outputs, dict):
            logit_key = model_params.get('logit_key', 'logits')
            logits = outputs[logit_key]
        else:
            logits = outputs
            
        loss_args = (outputs, labels)
        loss = loss_per_case_func(*loss_args, **loss_func_kwargs)
        loss = loss_agg_func(loss, **loss_agg_func_kwargs)

        global_step = tf.train.get_global_step()

        lr_func = lr_params.pop('func')
        learning_rate = lr_func(global_step=global_step, **lr_params)

        if mode == tf.estimator.ModeKeys.TRAIN:
            opt_func = opt_params.pop('optimizer', ClipOptimizer)
            # For deprecated parameter func
            old_opt_func = opt_params.pop('func', None)
            if old_opt_func:
                log.info('func in optimizer_params is deprecated, ' + \
                        'please use optimizer')
                opt_func = old_opt_func

            optimizer_base = opt_func(learning_rate=learning_rate, **opt_params)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer_base.minimize(loss, global_step)
        else:
            train_op = None


        eval_metrics = None
        if mode == tf.estimator.ModeKeys.EVAL:
           num_valid_targets = len(validation_params.keys())
           metric_fn_kwargs = {'labels': labels, 'logits': logits}
           if opt_params['use_tpu']:
               assert(num_valid_targets==1) # tpu estimators currently only support single targets :(
               first_valid = validation_params.keys()[0]
               valid_target = validation_params[first_valid]['targets']
               metric_fn = valid_target['func']
               if isinstance(outputs, dict):
                   for kw in outputs.keys():
                       if kw != logit_key:
                           kw_val = outputs[kw]
                           new_kw = kw
                           if isinstance(new_kw, int):
                               new_kw = 'i%i' % new_kw
                           metric_fn_kwargs.update({new_kw:kw_val})

               for kw in valid_target.keys():
                   v = valid_target[kw]
                   if isinstance(v, dict):
                       for kw1 in v.keys():
                           # add any additional kwargs
                           kw_val = v[kw1]
                           metric_fn_kwargs.update({kw1: kw_val})
                           #metric_fn_kwargs[kw] = kw_val
               eval_metrics = (metric_fn, metric_fn_kwargs)
           else:
               # normal estimators expect dicts and can support multiple targets (but same dataset and eval_steps etc)
               eval_dict = {}
               for k in validation_params.keys():
                   k_metric_fn_kwargs = metric_fn_kwargs
                   k_target = k['targets']
                   for kw in k_target.keys():
                       if kw != 'func':
                           # add any additional kwargs
                           kw_val = k_target[kw]
                           k_metric_fn_kwargs[kw] = kw_val                       
                   eval_dict[k] = (k_target['func'], k_metric_fn_kwargs)
               eval_metrics = eval_dict

        if opt_params['use_tpu']:
            return tpu_estimator.TPUEstimatorSpec(
              mode=mode,
              loss=loss,
              train_op=train_op,
              eval_metrics=eval_metrics)
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metrics)

    return model_fn, params_to_pass

def create_test_estimator_fn(use_tpu, 
                        model_params,
                        target_params):

    """
    Creates a model_fn for use with tf.Estimator for eval only. Uses predict mode.
    """
    # build params dictionary to be instantiated with the model_fn
    params_to_pass = {}
    params_to_pass['model_params'] = model_params
    params_to_pass['target_params'] = target_params
    params_to_pass['use_tpu'] = use_tpu

    def model_fn(features, labels, mode, params):
        model_params = params['model_params']
        target_params = params['target_params']
        model_params['train'] = (mode==tf.estimator.ModeKeys.TRAIN)
        if params['use_tpu']:
            model_params['batch_size'] = params['batch_size'] # per shard batch_size

        model_func = model_params.pop('func')
        outputs = model_func(inputs=features, **model_params)
        if isinstance(outputs, dict):
            logit_key = model_params.get('logit_key')
            if logit_key is None:
                logit_key = 'logits'
            logits = outputs[logit_key]
        else:
            logits = outputs

        predictions = None
        if mode == tf.estimator.ModeKeys.PREDICT:
           metric_fn_kwargs = {'labels': labels, 'logits': logits}

           eval_dict = {}
           k_metric_fn_kwargs = metric_fn_kwargs
           k_target = target_params['targets']
           for kw in k_target.keys():
               if kw != 'func':
                   # add any additional kwargs
                   kw_val = k_target[kw]
                   k_metric_fn_kwargs[kw] = kw_val   
           k_target_func = k_target.pop('func')                       
           eval_dict['predictions'] = k_target_func(**k_metric_fn_kwargs)
           predictions = eval_dict

        if params['use_tpu']:
            return tpu_estimator.TPUEstimatorSpec(
              mode=mode,
              predictions=predictions)
        else:
            return tf.Estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

    return model_fn, params_to_pass

def create_train_tpu_config(model_dir,
                      model_params,
                      tpu_name,
                      gcp_project,
                      steps_per_checkpoint,
                      tpu_zone=DEFAULT_TPU_ZONE,
                      num_shards=DEFAULT_NUM_SHARDS,
                      keep_checkpoint_max=5,
                      iterations_per_loop=DEFAULT_ITERATIONS_PER_LOOP):

    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=[tpu_name],
            zone=tpu_zone,
            project=gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    if iterations_per_loop == -1 or (steps_per_checkpoint is not None and steps_per_checkpoint < iterations_per_loop):
        log.info('Setting iterations_per_loop ({}) to be the same as steps_per_checkpoint ({}).'.format(iterations_per_loop, steps_per_checkpoint))
        iterations_per_loop = steps_per_checkpoint
        model_params['iterations_per_loop'] = iterations_per_loop

    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=model_dir,
        save_checkpoints_steps=steps_per_checkpoint,
        save_checkpoints_secs=None,
        keep_checkpoint_max=keep_checkpoint_max,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_shards))

    return config

def create_test_tpu_config(model_dir,
                      eval_steps,
                      tpu_name,
                      gcp_project,
                      tpu_zone=DEFAULT_TPU_ZONE,
                      num_shards=DEFAULT_NUM_SHARDS,
                      iterations_per_loop=DEFAULT_ITERATIONS_PER_LOOP):

    tpu_cluster_resolver = (
        tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu=[tpu_name],
            zone=tpu_zone,
            project=gcp_project))
    tpu_grpc_url = tpu_cluster_resolver.get_master()

    config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        evaluation_master=tpu_grpc_url,
        model_dir=model_dir,
        log_step_count_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=eval_steps,
            num_shards=num_shards))

    return config
