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


if 'TFUTILS_LOGFILE' in os.environ:
    logging.basicConfig(filename=os.environ['TFUTILS_LOGFILE'])
    print ("USING LOGFILE: %s" % os.environ['TFUTILS_LOGFILE'])
else:
    logging.basicConfig()
log = logging.getLogger('tfutils')
log.setLevel('DEBUG')

DEFAULT_HOST = '/cpu:0'
DEFAULT_DEVICES = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

DEFAULT_MODEL_SEED = 0
DEFAULT_LOOP_PARAMS = frozendict()
DEFAULT_DONT_RUN = False
DEFAULT_SKIP_CHECK = False
DEFAULT_LOG_DEVICE_PLACEMENT = False
DEFAULT_TRAIN_THRES_LOSS = 100
DEFAULT_LOAD_PARAMS = frozendict(
        {'do_restore': True, 
         'from_ckpt': None, 
         'to_restore': None, 
         'load_param_dict': None})
DEFAULT_LEARNING_RATE_PARAMS = frozendict({'func': tf.train.exponential_decay})

DEFAULT_LOSS_PARAMS = frozendict(
        {'pred_targets': ['labels'],
         'loss_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
         'agg_func': tf.reduce_mean})

DEFAULT_OPTIMIZER_PARAMS = frozendict(
        {'optimizer_class': tf.train.MomentumOptimizer,
         'momentum': 0.9})

DEFAULT_SAVE_PARAMS = frozendict({'save_metrics_freq': 100,
                                  'save_valid_freq': 3000,
                                  'cache_max_num': 6,
                                  'cache_filters_freq': 3000,
                                  'save_filters_freq': 30000,
                                  'save_initial_filters': True,
                                  'save_to_gfs': (),
                                  'do_save': True})

DEFAULT_PARAMS = frozendict({
    'dont_run': DEFAULT_DONT_RUN,
    'skip_check': DEFAULT_SKIP_CHECK,
    'model_params': frozendict(),
    'train_params': frozendict(),
    'validation_params': frozendict(),
    'log_device_placement': DEFAULT_LOG_DEVICE_PLACEMENT,
    'save_params': frozendict(DEFAULT_SAVE_PARAMS),
    'load_params': frozendict(DEFAULT_LOAD_PARAMS),
    'loss_params': frozendict(DEFAULT_LOSS_PARAMS),
    'optimizer_params': frozendict(DEFAULT_OPTIMIZER_PARAMS),
    'learning_rate_params': frozendict(DEFAULT_LEARNING_RATE_PARAMS),
})


def get_model_base(inputs, func, seed=0, train=False, **model_params):
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


def get_model(inputs, model_params, param=None, trarg=None):
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
    with tf.variable_scope(tf.get_variable_scope()):

        tower_outputs = []
        devices = model_params['devices']
        num_gpus = model_params['num_gpus']
        multi_gpu_inputs = split_input(inputs, num_gpus)
        # DEFAULT: Prepare loss and optimizer if training.
        if model_params['train']:
            assert param and trarg is not None

            tower_losses = []
            tower_grads = []

            (param['learning_rate_params'],
             learning_rate) = get_learning_rate(trarg['global_step'],
                                                **param['learning_rate_params'])
            (param['optimizer_params'],
             optimizer_base) = get_optimizer(learning_rate,
                                             param['optimizer_params'])

        # Distribute graph across desired devices.
        for device, each_input in zip(devices, multi_gpu_inputs):
            with tf.device(device), tf.name_scope('__GPU__' + device[-1]):

                model_params, output = get_model_base(each_input, **model_params)
                tower_outputs.append(output)

                tf.get_variable_scope().reuse_variables()

                # Get loss and optimizer if training
                if model_params['train']:
                    param['loss_params'], loss = get_loss(
                            each_input, output, 
                            **param['loss_params'])

                    tf.get_variable_scope().reuse_variables()

                    grad = optimizer_base.compute_gradients(loss)
                    tower_losses.append(loss)
                    tower_grads.append(grad)

    # Gather and aggregate outputs on the host (CPU).
    output = aggregate_outputs(tower_outputs)

    # DEFAULT: Accumulate and average gradients on the host (CPU).
    if model_params['train']:

        if param['train_params'].get('targets'):
            ttargs = copy.deepcopy(param['train_params']['targets'])
            ttargs_func = ttargs.pop('func')
            ttarg = ttargs_func(inputs, output, **ttargs)
            trarg['train_targets'].update(ttarg)

        # Aggregate loss.
        loss = tf.reduce_mean(tf.stack(tower_losses))

        # Aggregate and accumulate gradients.
        minibatch_grads = optimizer_base.aggregate_gradients(tower_grads)
        mini_flag, grads = optimizer_base.accumulate_gradients(
                minibatch_grads, 
                trarg['num_minibatches'])
        #grads = minibatch_grads

        # Apply accumulated gradients.
        optimizer = optimizer_base.apply_gradients(grads, trarg['global_step'])

        # Prepare train_targets
        if 'loss' not in trarg['train_targets']:
            trarg['train_targets']['loss'] = loss
        if '__grads__' not in trarg['train_targets']:
            trarg['train_targets']['__grads__'] = mini_flag
            pass
        if 'optimizer' not in trarg['train_targets']:
            trarg['train_targets']['optimizer'] = optimizer
        if 'learning_rate' not in trarg['train_targets']:
            trarg['train_targets']['learning_rate'] = learning_rate

        param['model_params'] = model_params
        return param['model_params'], output, param, trarg
    else:
        return model_params, output


def train_loop(sess, train_targets, num_minibatches=1, **loop_params):
    """Define default minibatch training loop.

    A training loop that performs minibatching with ``num_minibatches``
    minibatches.

    Args:
        sess (tf.Session): Current tensorflow session.
        train_targets (dict): Target operations to be evaluated by ``sess.run``.
            By default, ``base.train_from_params`` inserts the following
            targets to facilitate minibatching:
            * ``__grads__`` (tf.Operation): Accumulates and stores gradients.
            * ``optimizer`` (tf.Operation): Applies and zeros gradients.
        num_minibatches (int): number of minibatches to use.
        **loop_params (mapping): additional, user-defined kwargs to
            be used in the training loop.

    Returns:
        dict: A dictionary containing train targets evaluated by the session.

    """
    assert all([required in targets for targets in train_targets
                for required in ['__grads__', 'optimizer']])

    # Perform minibatching
    range_len = (int)(num_minibatches)
    for minibatch in range(range_len - 1):
        # Accumulate gradient for each minibatch
        sess.run([target['__grads__'] for target in train_targets])

    # Compute final targets (includes zeroing gradient accumulator variable)
    return sess.run(train_targets)


def get_learning_rate(global_step,
                      func=tf.train.exponential_decay,
                      **learning_rate_params):
    learning_rate = func(global_step=global_step,
                         **learning_rate_params)
    learning_rate_params['func'] = func
    return learning_rate_params, learning_rate


def get_optimizer(
        learning_rate,
        optimizer_params,
        ):
    if not optimizer_params:
        optimizer_params = dict(DEFAULT_PARAMS['optimizer_params'])
    builder = optimizer_params.pop('builder', ClipOptimizer)
    # For deprecated parameter func
    func = optimizer_params.pop('func', None)
    if func:
        log.info('func in optimizer_params is deprecated, ' + \
                'please use builder')
        builder = func

    # Build the optimizer, use class MinibatchOptimizer as a wrapper
    optimizer_base = MinibatchOptimizer(
            builder=builder,
            learning_rate=learning_rate,
            **optimizer_params)
    optimizer_params['builder'] = func
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


def get_loss_base(
        inputs,
        outputs,
        pred_targets,
        loss_func,
        loss_func_kwargs={},
        agg_func=None,
        agg_func_kwargs={},
        **loss_params):
    # Process some parameters
    loss_func_kwargs = copy.deepcopy(loss_func_kwargs)

    if not isinstance(pred_targets, (list, tuple, np.ndarray)):
        pred_targets = [pred_targets]
    pred_targets = list(pred_targets)

    # Get the labels to predict from inputs
    labels = [inputs[t] for t in pred_targets]
    try:
        # Usual way to call the loss function: 
        #   outputs will be sent as first parameter, labels will be unpacked
        loss = loss_func(outputs, *labels, **loss_func_kwargs)
    except:
        # Very special situation for 
        #   tf.nn.sparse_softmax_cross_entropy_with_logits,
        # which only accepts named parameters rather than positional parameters
        loss = loss_func(logits=outputs, labels=labels, **loss_func_kwargs)

    if agg_func:
        loss = agg_func(loss, **agg_func_kwargs)
    return loss


'''
Less important functions: 
'''


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
                queue_params = param.get(queue_params, None)
                assert not queue_params, \
                        "Queue methods are no longer supported!"\
                        + " Please use master_w_queue branch!"

                # Parse training data params (minibatching).
                if 'minibatch_size' not in param:
                    param['num_minibatches'] = 1
                    param['minibatch_size'] = param['data_params']['batch_size']
                    log.info('minibatch_size not specified for training data_params... ' +
                             'Defaulting minibatch_size to: {} (identical to the batch size).'
                             .format(param['data_params']['batch_size']))
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


def check_model_equivalence(m1, m2, name):
    """TODO: fill this in to make it stronger."""
    assert set(m1.keys()) == set(m2.keys()), (m1.keys(), m2.keys())


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
                     for now_arg in temp_args} for ind in xrange(n_gpus)]

    return list_of_args


def get_loss_dict(*args, **kwargs):
    kwargs = copy.copy(kwargs)
    name = kwargs.pop('name', 'loss')
    return {name: get_loss_base(*args, **kwargs)}
