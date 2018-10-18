from collections import OrderedDict
import tqdm
from tfutils.helper import \
        get_model, get_data, \
        get_loss_dict
import tfutils.utils as utils
import copy
import tensorflow as tf
from tfutils.defaults import DEFAULT_PARAMS, DEFAULT_LOOP_PARAMS


def get_validation_target(vinputs, voutputs,
                          agg_func=utils.identity_func,
                          online_agg_func=utils.append_and_return,
                          **validation_params):
    target_params = validation_params.get('targets', dict(DEFAULT_PARAMS['loss_params']))
    target_func = target_params.pop('func', get_loss_dict)
    vtargets = target_func(vinputs, voutputs, **target_params)
    target_params['func'] = target_func
    validation_params['targets'] = target_params

    valid_loop_params = validation_params.get('valid_loop', dict(DEFAULT_LOOP_PARAMS))
    valid_loop_func = valid_loop_params.pop('func', None)
    valid_loop = valid_loop_func
    valid_loop_params['func'] = valid_loop_func
    validation_params['valid_loop'] = valid_loop_params

    if 'num_steps' not in validation_params:
        assert hasattr(vinputs, 'total_batches'), '"num_batches" not specified in validation params, '\
            'data object must have "total_batches" attribute to be used as default.'
        validation_params['num_steps'] = vinputs.total_batches
    validation_params['agg_func'] = agg_func
    validation_params['online_agg_func'] = online_agg_func
    valid_targets = {'targets': vtargets,
                     'valid_loop': valid_loop,
                     'agg_func': validation_params['agg_func'],
                     'online_agg_func': validation_params['online_agg_func'],
                     'num_steps': validation_params['num_steps']}
    return validation_params, valid_targets


def get_valid_targets_dict(validation_params,
                           model_params,
                           loss_params,
                           cfg_final=None,
                           variable_m=None,
                           **params):
    """Helper function for creating validation target operations.

    NB: this function may modify validation_params.

    """
    valid_targets_dict = OrderedDict()
    model_params = copy.deepcopy(model_params)
    model_params['train'] = False
    prefix = model_params['prefix']
    if cfg_final is None:
        assert 'cfg_final' in model_params
        cfg_final = model_params['cfg_final']
    assert 'seed' in model_params
    for vtarg in validation_params:
        _, vinputs = get_data(**validation_params[vtarg]['data_params'])
        # scope_name = 'validation/%s' % vtarg
        scope_name = '{}/validation/{}'.format(prefix, vtarg)
        with tf.name_scope(scope_name):
            _mp, voutputs, variable_m = get_model(
                    vinputs, model_params,
                    variable_m=variable_m,
                    )
        validation_params[vtarg], valid_targets_dict[vtarg] = \
                get_validation_target(
                        vinputs, voutputs,
                        **validation_params[vtarg])

    return valid_targets_dict, variable_m


def run_all_validations(
        sess,
        targets,
        save_intermediate_freq=None,
        dbinterface=None,
        validation_only=False):
    """Helper function for actually computing validation results."""
    results = {}
    for target_name in targets:
        num_steps = targets[target_name]['num_steps']
        target = targets[target_name]['targets']
        agg_func = targets[target_name]['agg_func']
        online_agg_func = targets[target_name]['online_agg_func']
        valid_loop = targets[target_name]['valid_loop']
        results[target_name] = run_each_validation(
                sess,
                dbinterface,
                target_name,
                target,
                valid_loop,
                num_steps,
                online_agg_func,
                agg_func,
                save_intermediate_freq,
                validation_only)
    if dbinterface is not None:
        dbinterface.save(valid_res=results, validation_only=validation_only)
    return results


def run_each_validation(
        sess,
        dbinterface,
        target_name,
        target,
        valid_loop,
        num_steps,
        online_agg_func,
        agg_func,
        save_intermediate_freq=None,
        validation_only=False):
    """
    This function will run the validation for a number of steps.
    The results will be processed by online_agg_func for each step.
    And finally the result will be processed by agg_func
    """
    agg_res = None

    if save_intermediate_freq:
        prev_len = len(dbinterface.outrecs)

    # Run validation for each step
    for _step in tqdm.trange(num_steps, desc=target_name):
        if valid_loop:
            res = valid_loop(sess, target)
        else:
            res = sess.run(target)
        assert hasattr(res, 'keys'), 'result must be a dictionary'

        # Check whether should save
        if save_intermediate_freq \
                and _step % save_intermediate_freq == 0:
            dbinterface.save(valid_res={target_name: res},
                             step=_step,
                             validation_only=validation_only)

        # Process the results using online_agg_func
        agg_res = online_agg_func(agg_res, res, _step)

    # Get the final result using agg_func
    result = agg_func(agg_res)

    # Put results to database
    if save_intermediate_freq:
        dbinterface.sync_with_host()
        new_len = len(dbinterface.outrecs)
        result['intermediate_steps'] = dbinterface.outrecs[prev_len: new_len]

    return result
