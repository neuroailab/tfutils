Getting Started
===============

**"Hello World" - TFUtils style**

We will take our first steps with TFUtils by training the cononical *MNIST*
network and saving the results to a MongoDB database (The following assumes
you have successfully completed the installation process).

Configuring MongoDB
~~~~~~~~~~~~~~~~~~~

Before we can begin training MNIST, we must locate a running instance of
MongoDB and determine the port it is listening to (by default, MongoDB
is set to listen on port ``27017``).

To verify that an instance of MongoDB is running on your machine, type

.. code-block:: bash

    $ps aux | grep mongod

which should produce an output resembling

.. code-block:: bash

    root      8030  0.3  0.8 13326208376 1165680 ? Sl   Aug07 170:37 /usr/bin/mongod -f /etc/mongod.conf

Look for the lines:

::

    ...
    # network interfaces
    net:
      port: 29101
    ...

in the mongod configuration file (``/etc/mongod.conf`` above) to determine the port,

These tests show basic procedures for training, validating, and extracting features from
models.

Note about MongoDB:

The tests require a MongoDB instance to be available on the port defined by "testport" in
the code below. This db can either be local to where you run these tests (and therefore
on 'localhost' by default) or it can be running somewhere else and then by ssh-tunneled on
the relevant port to the host where you run these tests. [That is, before testing, you'd run

.. code-block:: bash

         ssh -f -N -L  [testport]:localhost:[testport] [username]@mongohost.xx.xx

on the machine where you're running these tests. ``[mongohost]`` is the where the mongodb
instance is running.

Specifying an Experiment
~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have a database ready to store our training results, let's specify
how they will be saved. Critically, any experiment executed by
TFUtils can be uniquely identified by the following 5-tuple:

+--------------+------------------------------------------+
| ``host``     | Hostname where database connection lives |
+--------------+------------------------------------------+
| ``port``     | Port where database connection lives     |
+--------------+------------------------------------------+
| ``dbname``   | Name of database for storage             |
+--------------+------------+-----------------------------+
| ``collname`` | Name of the database collection          |
+--------------+------------+-----------------------------+
| ``exp_id``   | Experiment id descriptor                 |
+--------------+------------------------------------------+

The variables host/port/dbname/coll/exp_id control the location of the saved
data for the run, in order of increasing specificity.

.. note::
  When choosing these, consider that:

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

With this in mind, let us specify our ``save_params``:

.. code-block:: python

    from tfutils import model, utils
    from tfutils.tests import mnist_data as data

    host = 'localhost'
    port = 27017  # default mongoDB port
    dbname = 'tfutils_database'
    collname = 'into_collection'
    exp_id = 'hello_world_training'

    params = {}
    params['save_params'] = {'host': host,
                             'port': port,
                             'dbname': dbname,
                             'collname': collname,
                             'exp_id': exp_id,
                             'save_valid_freq': 20,
                             'save_filters_freq': 200,
                             'cache_filters_freq': 100,
                             }

Defining a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

    params['model_params'] = {'func': model.mnist_tfutils}

    params['loss_params'] = {'targets': ['labels'],
                             'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
                             'agg_func': tf.reduce_mean})

    params['optimizer_params'] = {'optimizer_class': tf.train.MomentumOptimizer,
                                  'momentum': 0.9})

    params['train_params'] = {'data_params': {'func': data.build_data,
                                              'batch_size': 100,
                                              'group': 'train'},
                              'num_steps': 500
                              }
    params['learning_rate_params'] = {'learning_rate': 0.05,
                                      'decay_steps': num_batches_per_epoch,
                                      'decay_rate': 0.95,
                                      'staircase': True}

Including Validation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    params['validation_params'] = {'valid0': {'data_params': {'func': data.build_data,
                                                              'batch_size': 100,
                                                              'group': 'test'},
                                              'num_steps': 10,
                                              'agg_func': utils.mean_dict}}

Train!
~~~~~~

.. code-block:: python

    base.train_from_params(**params)
