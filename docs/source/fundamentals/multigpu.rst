Multi-GPU Training
------------------

To enable multi-gpu training using tfutils, you just need to specify the gpu devices you want to use in ``model_params``.
At that time, the ``batch_size`` specified in ``data_params`` will be the total batch size on all gpus.

For example, using the following ``model_params`` in previous training example will train mnist model using two gpus.

.. code-block:: python

    params['model_params'] = {
        'func': model.mnist_tfutils,
        'devices': ['/gpu:0', '/gpu:1']}
