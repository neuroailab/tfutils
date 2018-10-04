Multi-Model Training
--------------------

TFUtils supports training multiple models on different gpus using the same data, this can be extremely useful in hyperparameter explorations to reduce data loading overload from disks.
To utilize this function, you just need to pass a list of dictionaries to the major parameters, which need to include ``model_params`` and optionally include ``loss_params``, ``learning_rate_params``, and so on.

For example, using the following ``model_params`` in previous training example will train two mnist models at the same time independently on two gpus using the same data.

.. code-block:: python

    params['model_params'] = [
        {'func': model.mnist_tfutils},
        {'func': model.mnist_tfutils}]
