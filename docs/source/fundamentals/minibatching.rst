Minibatching
------------

TFUtils supports minibatch training, which means that the gradients will be aggregated over several small batches and then applied.
This is useful when you want to train the model using a bigger batch size, but your gpu memory cannot support that.
To enable this feature, you need to set parameter ``minibatch_size`` in ``train_params``.
After setting, ``batch_size`` will still be the effective batch size, which means the number of examples used for one gradient update and ``minibatch_size`` will be the number of examples passed through at the same time on gpu(s).
The gradients will be aggregated after ``batch_size / minibatch_size`` mini batches and then applied to the model.
