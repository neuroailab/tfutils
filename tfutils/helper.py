import argparse
from tfutils.utils import frozendict, format_devices
import os
import logging
import tensorflow as tf
import copy
import numpy as np
from tfutils.optimizer import ClipOptimizer, MinibatchOptimizer
import tfutils.utils as utils
from tfutils.utils import aggregate_outputs
import pdb
import inspect
from six import string_types

from tfutils.defaults import \
        BRANCH_QUEUE_NAME, DEFAULT_HOST, \
        DEFAULT_DEVICES, DEFAULT_MODEL_SEED, DEFAULT_DONT_RUN, \
        DEFAULT_SKIP_CHECK, DEFAULT_LOG_DEVICE_PLACEMENT, \
        DEFAULT_TRAIN_THRES_LOSS, DEFAULT_LOAD_PARAMS, \
        DEFAULT_LEARNING_RATE_PARAMS, DEFAULT_LOSS_PARAMS, \
        DEFAULT_OPTIMIZER_PARAMS, DEFAULT_SAVE_PARAMS, DEFAULT_PARAMS, \
        train_loop, mean_and_reg_loss
from tfutils.multi_gpu import easy_variable_mgr as variable_mgr


if 'TFUTILS_LOGFILE' in os.environ:
    logging.basicConfig(filename=os.environ['TFUTILS_LOGFILE'])
    print ("USING LOGFILE: %s" % os.environ['TFUTILS_LOGFILE'])
else:
    logging.basicConfig()
log = logging.getLogger('tfutils')
log.setLevel('DEBUG')


def get_model_base(inputs, func, seed=0, train=False, **old_model_params):
    model_params = copy.deepcopy(old_model_params)
    model_params['seed'] = seed
    model_params['train'] = train
    # Your model function should return:
    #   1. the outputs
    #   2. onfiguration files you want to put to database
    outputs, cfg_to_database = func(inputs=inputs,
                              **model_params)
    model_params['func'] = func
    model_params['cfg_to_database'] = cfg_to_database
    # For supporting previous behaviors
    model_params['cfg_final'] = cfg_to_database
    return model_params, outputs


def get_model(inputs, model_params, var_manager=None, param=None, trarg=None):
    """Return model and any other targets (loss + optimizer) specified.

    Args:
        inputs (tf.Operation): Model inputs
        model_params (dict): Specifies model configuration and must contain:
            'devices' (list): device specs (e.g. '/gpu:0')
            'train' (bool): whether getting model for training
        param (None, optional): Description.
        trarg (None, optional): Description.

    Returns:
        tuple: Description.

    """
    devices = model_params['devices']
    num_gpus = model_params['num_gpus']
    model_prefix = model_params['prefix']

    # var_manager is used for variable management in multiple gpu training
    if not var_manager:
        trainable_scopes = model_params['trainable_scopes']
        if isinstance(trainable_scopes, string_types):
            trainable_scopes = [trainable_scopes]
        if trainable_scopes is not None:
            trainable_scopes = [s.rstrip('/') for s in trainable_scopes]
            log.info('Only variables within the following scopes will be trained: {}'.format(trainable_scopes))
        var_manager = variable_mgr.VariableMgrLocalReplicated(
            model_prefix, devices,
            trainable_scopes=trainable_scopes)

    with tf.variable_scope(model_prefix):
        tower_outputs = []
        multi_gpu_inputs = split_input(inputs, num_gpus)

        # DEFAULT: Prepare loss and optimizer if training.
        if model_params['train']:
            assert param and trarg

            # Build global step
            trarg['global_step'] = tf.get_variable(
                    'global_step', [],
                    dtype=tf.int64, trainable=False,
                    initializer=tf.constant_initializer(0))

            # Lists that will include things from each gpu
            tower_losses = []
            tower_grads = []
            tower_opts = []
            opt_multi_mode = False

            param['learning_rate_params'], learning_rate \
                    = get_learning_rate(
                            trarg['global_step'],
                            **param['learning_rate_params'])

            # Make optimizers first, to avoid naming problems in optimizer
            for which_gpu, (device, each_input) \
                    in enumerate(zip(devices, multi_gpu_inputs)):
                with tf.device(device):
                    ## Build optimizer on each gpu
                    param['optimizer_params'], optimizer_base \
                            = get_optimizer(
                                    learning_rate,
                                    trarg['global_step'],
                                    param['train_params'].get('include_global_step'),
                                    param['optimizer_params'])
                    tower_opts.append(optimizer_base)
                    if hasattr(optimizer_base, '_multi_mode'):
                        if optimizer_base._multi_mode:
                            opt_multi_mode = True

        # Distribute graph across desired devices.
        update_ops = []
        for which_gpu, (device, each_input) \
                in enumerate(zip(devices, multi_gpu_inputs)):
            with var_manager.create_outer_variable_scope(which_gpu),\
                 tf.device(device), \
                 tf.name_scope('__GPU%i__' % (which_gpu)) as name_scope:

                new_model_params, output = get_model_base(
                        each_input,
                        **model_params)
                tower_outputs.append(output)

                tf.get_variable_scope().reuse_variables()

                # Get loss and gradients, collect update_ops if training
                if model_params['train']:
                    loss, grad, update_ops, param = get_loss_grad_updt(
                            param, each_input,
                            output, which_gpu,
                            update_ops, var_manager,
                            name_scope, tower_opts[which_gpu])
                    if isinstance(loss, list): # it means we are in multi-optimizer mode, so the intention is to sum this list
                        summed_loss = tf.add_n(loss)
                        tower_losses.append(summed_loss)
                    else:
                        tower_losses.append(loss)
                    tower_grads.append(grad)

        model_params = new_model_params
        tf.get_variable_scope().reuse_variables()

        # Gather and aggregate outputs on the host (CPU).
        output = aggregate_outputs(tower_outputs)

        # DEFAULT: Accumulate and average gradients on GPUs.
        if model_params['train']:
            trarg = get_train_targets(param, inputs, output, trarg)

            loss = tf.reduce_mean(tf.stack(tower_losses))
            with tf.variable_scope(variable_mgr.OPTIMIZER_NAME_SCOPE):
                mnb_accu_updt_list, optimizer_list = aggr_accu_apply_grads(
                        var_manager, trarg,
                        tower_grads, tower_opts, opt_multi_mode=opt_multi_mode)
            mnb_accu_updt_list = tf.group(*(mnb_accu_updt_list + update_ops))

            # Prepare train_targets
            trarg['train_targets']['loss'] = loss
            trarg['train_targets']['__grads__'] = mnb_accu_updt_list
            trarg['train_targets']['optimizer'] = optimizer_list
            trarg['train_targets']['learning_rate'] = learning_rate

            param['model_params'] = model_params
            return model_params, output, param, trarg, var_manager
        else:
            return model_params, output, var_manager


def get_learning_rate(global_step,
                      func=tf.train.exponential_decay,
                      **learning_rate_params):
    learning_rate = func(global_step=global_step,
                         **learning_rate_params)
    learning_rate_params['func'] = func
    return learning_rate_params, learning_rate


def get_optimizer(
        learning_rate,
        global_step,
        include_global_step,
        optimizer_params
        ):
    if not optimizer_params:
        optimizer_params = dict(DEFAULT_PARAMS['optimizer_params'])
    optimizer = optimizer_params.pop('optimizer', ClipOptimizer)
    # For deprecated parameter func
    func = optimizer_params.pop('func', None)
    if func:
        log.info('func in optimizer_params is deprecated, ' + \
                'please use optimizer')
        optimizer = func

    # Build the optimizer, use class MinibatchOptimizer as a wrapper
    if include_global_step:
        optimizer_base = MinibatchOptimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            global_step=global_step, 
            **optimizer_params)
    else:
        optimizer_base = MinibatchOptimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            **optimizer_params)
    optimizer_params['optimizer'] = optimizer
    return optimizer_params, optimizer_base


def get_data(func, **data_params):
    inputs = func(**data_params)
    data_params['func'] = func
    return data_params, inputs


def get_loss(train_inputs,
             train_outputs,
             pred_targets=DEFAULT_PARAMS['loss_params']['pred_targets'],
             agg_func=DEFAULT_PARAMS['loss_params']['agg_func'],
             loss_func=DEFAULT_PARAMS['loss_params']['loss_func'],
             targets=None,
             loss_per_case_func=None,
             **loss_params):
    # For previous parameters, now deprecated
    if loss_per_case_func:
        log.info('loss_per_case_func in loss_params is deprecated, ' + \
                'please use loss_func')
        loss_func = loss_per_case_func

    if targets is not None: # Given that [] is also False...
        log.info('targets in loss_params is deprecated, ' + \
                'please use pred_targets')
        pred_targets = targets

    # Set up the parameters, get actual loss
    loss_params['pred_targets'] = pred_targets
    loss_params['agg_func'] = agg_func
    loss_params['loss_func'] = loss_func
    loss = get_loss_base(train_inputs, train_outputs, **loss_params)
    return loss_params, loss


def get_train_targets(param, inputs, output, trarg):
    if param['train_params'].get('targets'):
        train_targts_args = copy.deepcopy(param['train_params']['targets'])
        train_targts_func = train_targts_args.pop('func')
        train_targets = train_targts_func(inputs, output, **train_targts_args)
        trarg['train_targets'].update(train_targets)
    reserved_keys = ['loss', '__grads__', 'optimizer', 'learning_rate']
    for each_key in reserved_keys:
        assert each_key not in trarg['train_targets'], \
                "Please avoid using reserved key %s in your targets!" \
                    % each_key
    return trarg


def get_loss_grad_updt(
        param, each_input, output, which_gpu,
        update_ops, var_manager,
        name_scope, curr_opt):
    param['loss_params']['which_device'] = which_gpu
    param['loss_params'], loss = get_loss(
            each_input, output,
            **param['loss_params'])

    update_ops.extend(
            tf.get_collection(
                tf.GraphKeys.UPDATE_OPS,
                name_scope))

    tf.get_variable_scope().reuse_variables()

    ## Get gradients for trainable vars on this gpu
    trainable_params = var_manager.trainable_variables_on_device(which_gpu)

    grad = curr_opt.compute_gradients(loss, var_list=trainable_params)
    return loss, grad, update_ops, param


def aggr_accu_apply_grads(var_manager, trarg, tower_grads, tower_opts, opt_multi_mode):
    # Aggregate and accumulate gradients.
    ## This is setting the devices where each gradient will be summed across
    ## all gpus
    apply_gradient_devices, gradient_state = (
            var_manager.preprocess_device_grads(tower_grads, opt_multi_mode=opt_multi_mode))

    ## mnb_accu_updt_list includes ops doing one minibatch,
    ## which includes accumulating gradients for this minibatch and
    ## also update_ops for this minibatch, which is usually for batch
    ## normalization
    mnb_accu_updt_list = []

    ## optimizer_list contains ops for applying gradients
    ## and global step updates
    gstep_update_op = tf.assign(
            trarg['global_step'],
            trarg['global_step']+1)
    optimizer_list = [gstep_update_op]

    ## Apply gradients on each gpu
    for d, device in enumerate(apply_gradient_devices):
        with var_manager.create_outer_variable_scope(d),\
             tf.device(device):
            avg_grads = var_manager.get_gradients_to_apply(
                    d, gradient_state)
            mnb_accu_grad, optimizer = tower_opts[d].accu_and_apply_grads(
                    avg_grads,
                    trarg['num_minibatches'],
                    None,
                    )

            mnb_accu_updt_list.append(mnb_accu_grad)
            optimizer_list.append(optimizer)
    return mnb_accu_updt_list, optimizer_list


def get_loss_base(
        inputs,
        outputs,
        pred_targets,
        loss_func,
        loss_func_kwargs={},
        agg_func=DEFAULT_PARAMS['loss_params']['agg_func'],
        agg_func_kwargs={},
        which_device=0,
        labels_to_dict=False,
        inputs_as_dict=False,
        **loss_params):
    # Process some parameters
    loss_func_kwargs = copy.deepcopy(loss_func_kwargs)

    if not inputs_as_dict:
        if not isinstance(pred_targets, (list, tuple, np.ndarray)):
            pred_targets = [pred_targets]
        pred_targets = list(pred_targets)

        # Get the labels to predict from inputs
        labels = [inputs[t] for t in pred_targets]
        loss_func_args = loss_func.__code__.co_varnames
        if '_sentinel' not in loss_func_args:
            # Usual way to call the loss function:
            #   outputs will be sent as first parameter, labels will be unpacked
            if labels_to_dict: 
                # put labels in dictionary for certain function signatures
                loss_func_kwargs['labels'] = [inputs[t] for t in pred_targets]
                labels = []
            loss = loss_func(outputs, *labels, **loss_func_kwargs)
        else:
            # Very special situation for
            #   tf.nn.sparse_softmax_cross_entropy_with_logits,
            # which only accepts named parameters rather than positional parameters
            assert len(labels)==1, \
                    'Should only have one thing to predict!'
            loss = loss_func(
                    logits=outputs,
                    labels=labels[0],
                    **loss_func_kwargs)
    else:
        loss = loss_func(outputs, inputs, **loss_func_kwargs)

    if not agg_func==mean_and_reg_loss:
        log.info('You are not using function mean_and_reg_loss provided in '\
                + 'tfutils.defaults, if you are using multi-gpu training, '\
                + 'please check that function to make sure your '\
                + 'regularization loss is multi-gpu safe!')
    if agg_func:
        # Check whether which_device is required for agg_func
        agg_func_args = agg_func.__code__.co_varnames
        if 'which_device' in agg_func_args:
            loss = agg_func(loss, which_device=which_device, **agg_func_kwargs)
        else:
            log.info('You are not requiring which_device parameter in your '\
                    + 'agg_func, if you are using multi-gpu training, '\
                    + 'please check function mean_and_reg_loss in '\
                    + 'tfutils.defaults to make sure your '\
                    + 'regularization loss is multi-gpu safe!')
            loss = agg_func(loss, **agg_func_kwargs)
    return loss


"""
Less important functions:
"""


def parse_params(mode,
                 model_params,
                 dont_run=False,
                 skip_check=False,
                 save_params=None,
                 train_params=None,
                 loss_params=None,
                 load_params=None,
                 optimizer_params=None,
                 validation_params=None,
                 learning_rate_params=None,
                 log_device_placement=False,
                 use_tpu=False
                 ):
    """Ensure the params dictionary has the correct structure.

    Each params arg must be a list of dictionaries where the ith element
    corresponds to parameters of the ith distinct model. Thus, the length of
    all params must be the same and reflect the number of distinct models
    to be evaluated.

    If an params arg does not satisfy the above requirements, ``parse_params``
    attempts to convert to the correct structure and logs any changes made.
    If it is missing any necessary components, defaults defined at the top of
    this module are used. If there exists an unresovlable conflict, an error
    is raised, and the user will be forced to resolve it before continuing.

    """
    model_params = [model_params] if not isinstance(model_params,
                                                    list) else model_params
    num_models = len(model_params)
    list_lens = [num_models]
    DEVICES = copy.copy(DEFAULT_DEVICES)

    params = {
        'dont_run': dont_run,
        'skip_check': skip_check,
        'load_params': load_params,
        'save_params': save_params,
        'model_params': model_params,
        'validation_params': validation_params,
        'log_device_placement': log_device_placement,
        }

    if mode == 'train':
        params.update({
            'loss_params': loss_params,
            'train_params': train_params,
            'optimizer_params': optimizer_params,
            'learning_rate_params': learning_rate_params})

    # Ensure params is a dict of lists, using defaults when necessary.
    for name, param_list in params.items():
        if not param_list and not isinstance(param_list, bool):
            if isinstance(DEFAULT_PARAMS[name], frozendict):
                param_list = dict(DEFAULT_PARAMS[name])
            else:
                param_list = DEFAULT_PARAMS[name]
        if not isinstance(param_list, list):
            param_list = [copy.deepcopy(param_list) for _ in range(num_models)]
        if len(param_list) != num_models and len(param_list) == 1:
            param_list += (num_models - 1) * copy.deepcopy(param_list)

        for model_num, param in enumerate(param_list):

            # Parse model_params.
            if name == 'model_params':
                if 'seed' not in param:
                    param['seed'] = DEFAULT_MODEL_SEED
                    log.info('No seed specified for model {}... '.format(model_num) +
                             'Defaulting to seed: {}.'.format(DEFAULT_MODEL_SEED))
                if 'prefix' not in param:
                    param['prefix'] = 'model_{}'.format(model_num)
                    log.info('No prefix specified for model {}... '.format(model_num) +
                             'Defaulting to prefix: {}.'.format(param['prefix']))
                if 'train' not in param:
                    if mode == 'train':
                        param['train'] = True
                    else:
                        param['train'] = False

                if 'trainable_scopes' not in param:
                    param['trainable_scopes'] = None
                    log.info('No trainable scopes specified for model {}... '.format(model_num) +
                             'All trainable variables will be trained by default.')

                if not use_tpu:
                    # Parse device specification.
                    if 'devices' not in param:
                        param['devices'] = [DEVICES.pop(0)]
                        log.info('No devices specified for model {}... '.format(model_num) +
                                 'Defaulting to gpus: {}.'.format(param['devices']))
                    param['devices'] = format_devices(param['devices'])

                    if 'num_gpus' not in param:
                        param['num_gpus'] = len(param['devices'])

                    if not isinstance(param['num_gpus'], list):
                        assert param['num_gpus'] == len(param['devices']), (
                           'num_gpus does not match the number of gpus specified in devices.')
                    else:
                        assert len(param['num_gpus']) == len(param['devices']), (
                           'num_gpus does not match the number of gpus specified in devices.')

            # Parse train_params.
            if name == 'train_params':
                if 'num_steps' not in param:
                    param['num_steps'] = DEFAULT_TRAIN_NUM_STEPS
                    log.info('num_steps not specified for model {}... '.format(model_num) +
                             'Defaulting num_steps to: {}.'.format(DEFAULT_TRAIN_NUM_STEPS))
                if 'thres_loss' not in param:
                    param['thres_loss'] = DEFAULT_TRAIN_THRES_LOSS
                    log.info('thres_loss not specified for model {}... '.format(model_num) +
                             'Defaulting thres_loss to: {}.'.format(DEFAULT_TRAIN_THRES_LOSS))
                if 'train_loop' not in param:
                    param['train_loop'] = {'func': train_loop}
                    log.info('train_loop not specified for model {}... '.format(model_num) +
                             'Using default training loop.')
                if 'validate_first' not in param:
                    param['validate_first'] = True
                    log.info('validate_fist not specified for model {}... '.format(model_num) +
                             'Defaulting validate_first to: {}.'.format(param['validate_first']))

                # If queue_params found, users should use old master
                queue_params = param.get('queue_params', None)
                assert not queue_params, \
                        "Queue methods are no longer supported!"\
                        + " Please use %s branch!" % BRANCH_QUEUE_NAME

                # Parse training data params (minibatching).
                if 'minibatch_size' not in param:
                    param['num_minibatches'] = 1
                    if not use_tpu:
                        param['minibatch_size'] = param['data_params']['batch_size']
                        log.info('minibatch_size not specified for training data_params... ' +
                                 'Defaulting minibatch_size to: {} (identical to the batch size).'
                                 .format(param['data_params']['batch_size']))
                else:
                    if use_tpu:
                        batch_size = param['train_params']['batch_size']
                        minibatch_size = param['minibatch_size']
                        if batch_size != minibatch_size:
                            log.info('Minibatching currently not supported on TPU. Setting minibatch size ({}) ' +
                                     'to be original batch size ({}).'.format(minibatch_size, batch_size))
                        param['minibatch_size'] = batch_size
                        param['num_minibatches'] = 1
                    else:
                        batch_size = param['data_params']['batch_size']
                        minibatch_size = param['minibatch_size']
                        assert minibatch_size <= batch_size, (
                               'Minibatch size cannot be larger than batch size.')

                        num_minibatches = batch_size / float(minibatch_size)
                        if num_minibatches % 1 != 0:
                            num_minibatches = round(num_minibatches)
                            minibatch_size = batch_size / num_minibatches
                            log.warning('Minibatch size ({}) does not divide batch size ({}) evenly...'
                                        .format(minibatch_size, batch_size))
                            log.info('Rounding minibatch size to closest factor of batch size: {}'
                                     .format(minibatch_size))
                        param['minibatch_size'] = minibatch_size
                        param['num_minibatches'] = num_minibatches
                        param['data_params']['batch_size'] = minibatch_size

        params[name] = param_list

        list_lens.append(len(param_list))
        assert isinstance(param_list, list), '{} should also be a list'.format(name)
        assert len(param_list) == num_models, '{} should have length'.format(num_models)
    assert len(np.unique(list_lens)) == 1, 'All param lists should have be same length!'

    # Append the model_prefix to non-unique exp_ids.
    for key in ['save_params', 'load_params']:
        unique_exp_ids = set(s.get('exp_id') for s in params[key])
        if None not in unique_exp_ids:
            if len(unique_exp_ids) == 1 and num_models != 1:
                log.warning('Non-unique exp_ids detected in {} '.format(key) +
                            'with multiple models.'.format(key))
                for i, (p, mp) in enumerate(zip(params[key],
                                                params['model_params'])):
                    p.update({'exp_id': p.get('exp_id') + '_' + mp['prefix']})
                    log.info('Appending \'_{} to the exp_id of model number {}.'.
                             format(mp['prefix'], i))
                    log.info('New exp_id is: {}'.format(p.get('exp_id')))

            assert len(set(s['exp_id'] for s in params[key])) == num_models

    # Prepare run_args to be passed to `base.(train|test)(**run_args)`.
    run_args = {
        'dbinterface': num_models * [None],
        'validation_targets': [dict() for _ in range(num_models)]}

    if mode == 'test':
        run_args.update({
            'save_intermediate_freq': num_models * [None]})
    else:
        run_args.update({
            'global_step': num_models * [None],
            'train_targets': [dict() for _ in range(num_models)],
            'num_steps': [p['num_steps'] for p in params['train_params']],
            'thres_loss': [p['thres_loss'] for p in params['train_params']],
            'train_loop': [p['train_loop']['func'] for p in params['train_params']],
            'validate_first': [p['validate_first'] for p in params['train_params']],
            'num_minibatches': [p['num_minibatches'] for p in params['train_params']]})

    return params, run_args


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', type=json.loads, default=None)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    for p in filter(lambda x: x.endswith('_func'), args):
        modname, objname = args[p].rsplit('.', 1)
        mod = importlib.import_module(modname)
        args[p] = getattr(mod, objname)
    return args


def split_input(inputs, num_gpus=1):
    if not isinstance(num_gpus, list):
        n_gpus = num_gpus
    else:
        n_gpus = len(num_gpus)

    if n_gpus == 1:
        return [inputs]

    temp_args = {v: tf.split(inputs[v], axis=0, num_or_size_splits=num_gpus)
                 for v in inputs}

    list_of_args = [{now_arg: temp_args[now_arg][ind]
                     for now_arg in temp_args} for ind in range(n_gpus)]

    return list_of_args


def get_loss_dict(*args, **kwargs):
    kwargs = copy.copy(kwargs)
    name = kwargs.pop('name', 'loss')
    return {name: get_loss_base(*args, **kwargs)}
