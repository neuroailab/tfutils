from __future__ import absolute_import, division, print_function

import os
import re
import sys
import time
import importlib
import json
import copy

from bson.objectid import ObjectId
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.ops import variables
import numpy as np

import tfutils.utils as utils
from tfutils.optimizer import ClipOptimizer
from tfutils.error import HiLossError, NoGlobalStepError, NoChangeError
from tfutils.utils import strip_prefix, aggregate_outputs
from tfutils.db_interface import DBInterface
from tfutils.helper import \
        parse_params, get_params, \
        DEFAULT_PARAMS, DEFAULT_TRAIN_THRES_LOSS, \
        split_input, log
from tfutils.validation import run_all_validations, get_valid_targets_dict

DEFAULT_HOST = '/cpu:0'
DEFAULT_LOOP_PARAMS = frozendict()


def train_from_params(
        save_params,
        model_params,
        train_params,
        loss_params=None,
        learning_rate_params=None,
        optimizer_params=None,
        validation_params=None,
        load_params=None,
        log_device_placement=DEFAULT_PARAMS['log_device_placement'], # advanced
        dont_run=DEFAULT_PARAMS['dont_run'], # advanced
        skip_check=DEFAULT_PARAMS['skip_check'], # advanced
        ):
    """
    Main training interface function.

    Args:
        save_params (dict): 
            Describing the parameters need to construct the save database, and
            control saving. These include:
            - host (str)
                Hostname where database connection lives
            - port (int)
                Port where database connection lives
            - dbname (str)
                Name of database for storage
            - collname (str)
                Name of collection for storage
            - exp_id (str)
                Experiment id descriptor
                NOTE: the variables host/port/dbname/coll/exp_id control
                the location of the saved data for the run, in order of
                increasing specificity.  When choosing these, note that:
                    1.  If a given host/port/dbname/coll/exp_id already has saved checkpoints,
                        then any new call to start training with these same location variables
                        will start to train from the most recent saved checkpoint.  If you mistakenly
                        try to start training a new model with different variable names, or structure,
                        from that existing checkpoint, an error will be raised, as the model will be
                        incompatiable with the saved variables.
                    2.  When choosing what dbname, coll, and exp_id, to use, keep in mind that mongodb
                        queries only operate over a single collection.  So if you want to analyze
                        results from a bunch of experiments together using mongod queries, you should
                        put them all in the same collection, but with different exp_ids.  If, on the
                        other hand, you never expect to analyze data from two experiments together,
                        you can put them in different collections or different databases.  Choosing
                        between putting two experiments in two collections in the same database
                        or in two totally different databases will depend on how you want to organize
                        your results and is really a matter of preference.
            - do_save (bool, default: True)
                Whether to save to database
            - save_initial_filters (bool, default: True)
                Whether to save initial model filters at step = 0,
            - save_metrics_freq (int, default: 5)
                How often to store train results to database
            - save_valid_freq (int, default: 3000)
                How often to calculate and store validation results
                                        to database
            - save_filters_freq (int, default: 30000)
                How often to save filter values to database
            - cache_filters_freq (int, default: 3000)
                How often to cache filter values locally and save
                to ___RECENT database
            - cache_max_num (int, default: 6)
                Maximal number of cached filters to keep in __RECENT database
            - cache_dir (str, default: None)
                Path where caches will be saved locally. If None, will default to
                ~/.tfutils/<host:post>/<dbname>/<collname>/<exp_id>.

        model_params (dict): Containing function that produces model and arguments to that function.
            - model_params['func'] is the function producing the model.

                The function's signature is:
                ::

                    inputs: data object
                    - ``train`` -- boolean if training is happening
                    - ``seed`` -- seed for use in random generation of final config

                Returns:
                - outputs (tf.Operations): train output tensorflow nodes
                - final configuration used in model

            - Remaining items in model_params are dictionary of arguments passed to func.

        train_params (dict): Containing params for data sources and targets in training.

            - ``train_params['data_params']`` contains params for the data
                - ``train_params['data_params']['func']`` is the function that constructs the data:

                    The function's signature is:
                    ::

                        inputs: 
                        - ``batch_size`` -- Batch size for input data

                    Returns:
                    - inputs (tf.Operations): this will be sent to model function

                - ``train_params['data_params']['batch_size']`` batch size of the data, will be sent to func

                - Remainder of ``train_params['data_params']`` are kwargs passed to func

            - ``train_params['targets']`` (optional) contains params for additional train targets

                - ``train_params['targets']['func']`` is a function that produces
                  tensorflow nodes as training targets:

                    The function's signature is:
                    ::

                        inputs: 
                        - ``inputs`` -- returned values of ``train_params['data_params']['func']``
                        - ``output`` -- first returned value of ``train_params['model_params']['func']``

                    Returns:
                    A dictionary of tensors that will be computed and stored in the database

                - Remainder of ``train_parms['targets']`` are arguments to func.

            - ``train_params['validate_first']`` (optional, bool, default is True):
                controls whether validating before training

            - ``train_params['thres_loss']`` (optional, float, default: 100): 
                If loss exceeds this during training, HiLossError is thrown

            - ``train_params['num_steps']`` (int or None, default: None): 
                How many total steps of the optimization are run.
                If None, train is run until process is cancelled.

        loss_params (dict): Parameters for to utils.get_loss function for specifying loss.

            - ``loss_params['targets']` is a string or a list of strings,
              contain the names of inputs nodes that will be sent into the loss function

            - ``loss_params['loss_per_case_func']`` is the function used to calculate the loss.
              Must be provided. The parameters sent to this function is defined by loss_params['loss_per_case_func_params'].

            - ``loss_params['loss_per_case_func_params']`` is a dict including  help information about
              how positional parameters should be sent to loss_params['loss_per_case_func'] as named parameters.
              Default is ``{'_outputs': 'logits', '_targets_': 'labels'}``

            - If ``loss_params['loss_per_case_func_params']`` is empty, the parameters for
              loss_params['loss_per_case_func'] will be (outputs, *[inputs[t] for t in targets], **loss_func_kwargs),
              where 'outputs' is the output of the network, inputs is the input nodes,
              and targets is ``loss_params['targets']``.

            Key value can have three choices:
            - '_outputs': the value of this key will be the name for 'outputs'.
            - '_targets_': name for '[inputs[t] for t in targets]'.
            - '_target_somename': name for 'inputs[somename]' is somename is inside targets.

        - Parameters not mentioned by the key values will still be sent to the function as positional parameters.
            - ``loss_params['agg_func']`` is the aggregate function, default is None
            - ``loss_params['loss_func_kwargs']``. Keyword parameters sent to loss_params['loss_per_case_func']. Default is None.
            - ``loss_params['agg_func_kwargs']`. Keyword parameters sent to ``loss_params['agg_func']. Default is None.

        learning_rate_params (dict): Parameters for specifying learning_rate.
                - :obj:`learning_rate_params['func']` is a function producing
                  tensorflow node acting as learning rate. This function must accept argument "global_step".
                - remainder of learning_rate_params are arguments to func.

        optimizer_params (dict): Parameters for creating optimizer.
            - optimizer_params['func'] is a function producing a
              tensorflow optimizer object (like a subclass of tf.train.Optimizer)

            Must accept:
            - "learning_rate" -- the result of the learning_rate_func call
            - Must return object with a method called "minimize" with
              the same call signature as tensorflow.train.Optimizer.minimize --- that is:
            - Must accept:
                * "loss" -- result of loss_func call
                * "global_step" -- global step used for determine learning rate,
            Must return:
                * tensorflow node which computes gradients and applies
                  them, and must increment "global_step"
            - Remainder of optimizer_params (aside form "func") are arguments
              to the optimizer func

        validation_params (dict): Dictionary of validation sources. The structure if this dictionary is:

            {
                <validation_target_name_1>: {
                    'data': {
                        'func': (callable) data source function for this validation,
                        <kwarg1>: <value1> for 'func',
                        ...
                        },
                    'targets': {
                        'func': (callable) returning targets,
                        <kwarg1>: <value1> for 'func',
                        ...
                        }
                    'num_steps': (int) number of batches of validation source to compute
                    'agg_func': (optional, callable) how to aggregate validation results
                            across batches after computation. Signature is:
                                - one input argument: the list of validation batch results
                                - one output: aggregated version
                            Default is utils.identity_func
                    'online_agg_func': (optional, callable) how to aggregate validation results
                            on a per-batch basis. Siganture is:
                                - three input arguments: (current aggregate, new result, step)
                                - one output: new aggregated result
                            One first step, current aggregate passed in is None.
                            The final result is passed to the "agg_func".
                            Default is utils.append_and_return
                },
                <validation_target_name_2>: ...
            }

            For each validation_target_name key, the targets are computed and then added to
            the output dictionary to be computed every so often -- unlike train_targets which
            are computed on each time step, these are computed on a basic controlled by the
            valid_save_freq specific in the save_params.

        load_params (dict):
            Similar to save_params, if you want loading to happen from a different
            location than where saving occurs. Parameters include:
            - host (str)
                Hostname where database connection lives
            - port (int)
                Port where database connection lives
            - dbname (str)
                Name of database for storage
            - collname (str)
                Name of collection for storage
            - exp_id (str)
                Experiment id descriptor
            - do_restore (bool, default: True)
                Whether to restore from saved model
            - load_query (dict)
                mongodb query describing how to load from loading database
            - from_ckpt (string)
                Path to load from a TensorFlow checkpoint (instead of from the db)
            - to_restore (list of strings or a regex/callable which returns strings)
                Specifies which variables should be loaded from the checkpoint.
                Any variables not specified here will be reinitialized.
            - load_param_dict (dict)
                A dictionary whose keys are the names of the variables that are to be loaded
                from the checkpoint, and the values are the names of the variables of the model
                that you want to restore with the value of the corresponding checkpoint variable.

        log_device_placement (bool, default: False): Whether to log device placement in tensorflow session
        dont_run (bool, default: False): Whether returning everything, not actually training 
        skip_check (bool, default: False): Whether skipping github check, could be useful when working in detached head

    Returns:
        TYPE: Description.

    """
    params, train_args = parse_params('train',
                                      model_params,
                                      dont_run=dont_run,
                                      skip_check=skip_check,
                                      load_params=load_params,
                                      loss_params=loss_params,
                                      save_params=save_params,
                                      train_params=train_params,
                                      optimizer_params=optimizer_params,
                                      validation_params=validation_params,
                                      learning_rate_params=learning_rate_params,
                                      log_device_placement=log_device_placement,
                                      )

    with tf.Graph().as_default(), tf.device(DEFAULT_HOST):
        # For convenience, use list of dicts instead of dict of lists
        _params = [{key: value[i] for (key, value) in params.items()}
                   for i in range(len(params['model_params']))]
        _trargs = [{key: value[i] for (key, value) in train_args.items()}
                   for i in range(len(params['model_params']))]

        # Use a single dataprovider for all models.
        data_params = _params[0]['train_params']['data_params']

        _params[0]['train_params']['data_params'], inputs = get_data(**data_params)

        # Build a graph for each distinct model.
        for param, trarg in zip(_params, _trargs):
            with tf.variable_scope(param['model_params']['prefix']):
                # Build global step
                trarg['global_step'] = tf.get_variable('global_step', [],
                                                       dtype=tf.int64, trainable=False,
                                                       initializer=tf.constant_initializer(0))

                _, _, param, trarg = get_model(inputs,
                                               param['model_params'],
                                               param=param,
                                               trarg=trarg)

                tf.get_variable_scope().reuse_variables()

                trarg['validation_targets'] = \
                        get_valid_targets_dict(
                                **param)

        # Create session.
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    gpu_options=gpu_options,
                    log_device_placement=log_device_placement,
                    ))

        init_op_global = tf.global_variables_initializer()
        sess.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        sess.run(init_op_local)
        log.info('Initialized from scratch first')

        for param, trarg in zip(_params, _trargs):

            prefix = param['model_params']['prefix'] + '/'
            all_vars = variables._all_saveable_objects()
            var_list = strip_prefix(prefix, all_vars)
            for var in var_list:
                print(var)

            trarg['dbinterface'] = DBInterface(sess=sess,
                                               params=param,
                                               var_list=var_list,
                                               global_step=trarg['global_step'],
                                               save_params=param['save_params'],
                                               load_params=param['load_params'])
            trarg['dbinterface'].initialize()

        # Convert back to a dictionary of lists
        params = {key: [param[key] for param in _params]
                  for key in _params[0].keys()}
        train_args = {key: [trarg[key] for trarg in _trargs]
                      for key in _trargs[0].keys()}

        if dont_run:
            return train_args

        return train(sess, **train_args)


def train(sess,
          dbinterface,
          train_loop,
          train_targets,
          global_step,
          num_minibatches=1,
          num_steps=float('inf'),
          thres_loss=DEFAULT_TRAIN_THRES_LOSS,
          validate_first=True,
          validation_targets=None):
    """Actually runs the training evaluation loop.

    Args:
        sess (tesorflow.Session):
            Object in which to run calculations.

        dbinterface (DBInterface object): Saver through which to save results.

        train_loop (callable withs args: sess and train_targets):
            Callable that specifies a custom training loop
        train_targets (dict of tensorflow nodes): Targets to train.
            One item in this dict must be "optimizer" or similar
            to make anything happen
        num_minibatches (int): How many minibatches to use to before applying gradient update.
        num_steps (int): How many steps to train to before quitting
        validation_targets (dict of tensorflow objects, default: None):
            Objects on which validation will be computed
        thres_loss (float, default: 100):
            If loss exceeds this during training, HiLossError is thrown

    """
    # Collect args in a dict of lists
    train_args = {
        'num_steps': num_steps,
        'thres_loss': thres_loss,
        'train_loop': train_loop,
        'global_step': global_step,
        'dbinterface': dbinterface,
        'train_targets': train_targets,
        'validate_first': validate_first,
        'num_minibatches': num_minibatches,
        'validation_targets': validation_targets}

    # Convert to a list of dicts
    trargs = [{key: value[i] for (key, value) in train_args.items()}
              for i in range(len(train_targets))]

    num_steps = [t['num_steps'] for t in trargs]
    steps = [t['global_step'].eval(session=sess) for t in trargs]

    # Start initial validation
    for (step, trarg) in zip(steps, trargs):

        if step >= trarg['num_steps']:
            log.info('Training cancelled since step ({}) is >= num_steps ({})'.
                     format(step, trarg['num_steps']))
            return

        log.info('Training beginning ...')

        if step == 0:
            trarg['dbinterface'].start_time_step = time.time()
            if trarg['validate_first']:
                valid_res = run_all_validations(
                        sess,
                        trarg['validation_targets'],
                        dbinterface=trarg['dbinterface'])
    train_loop = train_args['train_loop'][0]
    train_targets = train_args['train_targets']

    # Run training
    while any(step < num_step for (step, num_step) in zip(steps, num_steps)):

        start_time_step = time.time()
        train_results = train_loop(sess, train_targets, num_minibatches=trarg['num_minibatches'])

        for (step, trarg, train_res) in zip(steps, trargs, train_results):

            old_step = step
            step = trarg['global_step'].eval(session=sess)

            if step <= old_step:
                raise NoChangeError('Your optimizer should have incremented the global step,'
                                    ' but did not: old_step=%d, new_step=%d' % (old_step, step))
            if train_res['loss'] > trarg['thres_loss']:
                raise HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_res['loss'],
                                                                                     trarg['thres_loss']))

            # Validation
            vtargs = trarg['validation_targets'] if step % trarg['dbinterface'].save_valid_freq == 0 else {}
            valid_res = run_all_validations(sess, vtargs)

            # Save
            trarg['dbinterface'].start_time_step = start_time_step
            trarg['dbinterface'].save(train_res=train_res,
                                      valid_res=valid_res,
                                      validation_only=False)

        steps = [t['global_step'].eval(session=sess) for t in trargs]

    # Sync and close the session
    res = []
    for trarg in trargs:
        trarg['dbinterface'].sync_with_host()
        res.append(trarg['dbinterface'].outrecs)

    sess.close()
    return res


def get_data(func, **data_params):
    inputs = func(**data_params)
    data_params['func'] = func
    return data_params, inputs


def get_model_base(input, func, seed=0, train=False, **model_params):
    model_params['seed'] = seed
    model_params['train'] = train
    outputs, cfg_final = func(inputs=input,
                              **model_params)
    model_params['func'] = func
    model_params['cfg_final'] = cfg_final
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
        mini_flag, grads = optimizer_base.accumulate_gradients(minibatch_grads, trarg['num_minibatches'])
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


def get_loss(train_inputs,
             train_outputs,
             targets=DEFAULT_PARAMS['loss_params']['targets'],
             agg_func=DEFAULT_PARAMS['loss_params']['agg_func'],
             loss_per_case_func=DEFAULT_PARAMS['loss_params']['loss_per_case_func'],
             **loss_params):
    loss_params['targets'] = targets
    loss_params['agg_func'] = agg_func
    loss_params['loss_per_case_func'] = loss_per_case_func
    loss = utils.get_loss(train_inputs, train_outputs, **loss_params)
    return loss_params, loss


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
    func = optimizer_params.pop('func', ClipOptimizer)
    optimizer_base = func(learning_rate=learning_rate,
                          **optimizer_params)
    optimizer_params['func'] = func
    return optimizer_params, optimizer_base


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
