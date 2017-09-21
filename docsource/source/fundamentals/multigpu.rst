Multi-GPU Training
------------------

.. code-block:: python

    params['model_params'] = {
        'func': model.mnist_tfutils,
        'devices': ['/gpu:0', '/gpu:1']}