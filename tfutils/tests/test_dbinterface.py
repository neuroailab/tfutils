"""Test DBInterface."""

import os
import re
import sys
import time
import errno
import shutil
import logging
import pymongo
import unittest
import pdb

import tensorflow as tf
import mnist_data as data

sys.path.insert(0, "..")

import tfutils.base as base
import tfutils.model as model
import tfutils.optimizer as optimizer
from tfutils.db_interface import TFUTILS_HOME
from tfutils.db_interface import DBInterface


# def logPoint(context):
#     """Utility function used for module functions and class methods."""
#     callingFunction = inspect.stack()[1][3]
#     print 'in %s - %s()' % (context, callingFunction)


def setUpModule():
    """Set up module once, before any TestCases are run."""
    logging.basicConfig()
    # logPoint('module %s' % __name__)


def tearDownModule():
    """Tear down module after all TestCases are run."""
    pass
    # logPoint('module %s' % __name__)


class TestDBInterface(unittest.TestCase):

    PORT = 29101
    HOST = 'localhost'
    EXP_ID = 'TEST_EXP_ID'
    DATABASE_NAME = 'TFUTILS_TESTDB'
    COLLECTION_NAME = 'TFUTILS_TESTCOL'
    CACHE_DIR = 'TFUTILS_TEST_CACHE_DIR'

    @classmethod
    def setUpClass(cls):
        """Set up class once before any test methods are run."""
        cls.setup_log()
        cls.setup_conn()
        cls.setup_cache()
        cls.setup_params()

    @classmethod
    def tearDownClass(cls):
        """Tear down class after all test methods have run."""
        cls.remove_directory(cls.CACHE_DIR)
        cls.remove_database(cls.DATABASE_NAME)

        # Close primary MongoDB connection.
        cls.conn.close()

    def setUp(self):
        """Set up class before _each_ test method is executed.

        Creates a tensorflow session and instantiates a dbinterface.

        """
        self.setup_model()
        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=tf.GPUOptions(allow_growth=True),
                log_device_placement=self.params['log_device_placement'],
                ))

        # TODO: Determine whether this should be called here or
        # in dbinterface.initialize()
        self.sess.run(tf.global_variables_initializer())

        self.dbinterface = DBInterface(
                sess=self.sess,
                params=self.params,
                cache_dir=self.CACHE_DIR,
                save_params=self.save_params,
                load_params=self.load_params)

        self.step = 0

    def tearDown(self):
        """Tear Down is called after _each_ test method is executed."""
        self.sess.close()

    @unittest.skip("skipping")
    def test_init(self):
        # TODO: Test all permutations of __init__ params.
        pass

    @unittest.skip("skipping")
    def test_load_rec(self):
        pass

    @unittest.skip("skipping")
    def test_initialize(self):
        pass

    def test_get_restore_vars(self):

        # First, train model and save a checkpoint
        self.train_model()  # weights_name='Weights'
        saved_path = self.save_test_checkpoint()

        # Create a new model with different variable names.
        self.setup_model(weights_name='Filters')

        # Reset var_list in DBInterface
        self.dbinterface.var_list = {
                var.op.name: var for var in tf.global_variables()}

        # Restore first checkpoint vars.
        mapping = {'Weights': 'Filters'}
        self.dbinterface.load_param_dict = mapping
        restore_vars = self.dbinterface.get_restore_vars(saved_path)

        self.log.info('restore_vars:')
        for name, var in restore_vars.items():
            if name in mapping.keys():
                self.log.info('(name, var.name): ({}, {})'.format(name, var.name))
                self.assertEqual(var.op.name, mapping[name])

    def test_filter_var_list(self):

        var_list = {var.op.name: var for var in tf.global_variables()}

        # Test None
        self.dbinterface.to_restore = None
        filtered_var_list = self.dbinterface.filter_var_list(var_list)
        self.assertEqual(filtered_var_list, var_list)

        # Test list of strings
        self.dbinterface.to_restore = ['Weights']
        filtered_var_list = self.dbinterface.filter_var_list(var_list)
        for name, var in filtered_var_list.items():
            self.assertIn(name, ['Weights'])
            self.assertNotIn(name, ['Bias', 'global_step'])

        # Test regex
        self.dbinterface.to_restore = re.compile(r'Bias')
        filtered_var_list = self.dbinterface.filter_var_list(var_list)
        for name, var in filtered_var_list.items():
            self.assertIn(name, ['Bias'])
            self.assertNotIn(name, ['Weights', 'global_step'])

        # Test invalid type (should raise TypeError)
        self.dbinterface.to_restore = {'invalid_key': 'invalid_value'}
        with self.assertRaises(TypeError):
            filtered_var_list = self.dbinterface.filter_var_list(var_list)

    @unittest.skip("skipping")
    def test_tf_saver(self):
        pass

    @unittest.skip("skipping")
    def test_load_from_db(self):
        pass

    @unittest.skip("skipping")
    def test_save(self):
        self.dbinterface.initialize()
        self.dbinterface.start_time_step = time.time()
        train_res = self.train_model(num_steps=100)
        self.dbinterface.save(train_res=train_res, step=self.step)

    @unittest.skip("skipping")
    def test_sync_with_host(self):
        pass

    @unittest.skip("skipping")
    def test_save_thread(self):
        pass

    @unittest.skip("skipping")
    def test_initialize_from_ckpt(self):
        save_path = self.save_test_checkpoint()
        self.load_test_checkpoint(save_path)

    def train_model(self, num_steps=100):
        x_train = [1, 2, 3, 4]
        y_train = [0, -1, -2, -3]
        x = tf.get_default_graph().get_tensor_by_name('x:0')
        y = tf.get_default_graph().get_tensor_by_name('y:0')
        feed_dict = {x: x_train, y: y_train}

        pre_global_step = self.sess.run(self.global_step)
        for step in range(num_steps):
            train_res = self.sess.run(self.train_targets, feed_dict=feed_dict)
            self.log.info('Step: {}, loss: {}'.format(step, train_res['loss']))

        post_global_step = self.sess.run(self.global_step)
        self.assertEqual(pre_global_step + num_steps, post_global_step)
        self.step += num_steps
        return train_res

    def save_test_checkpoint(self):
        self.log.info('Saving checkpoint to {}'.format(self.save_path))
        saved_checkpoint_path = self.dbinterface.tf_saver.save(self.sess,
                                                               save_path=self.save_path,
                                                               global_step=self.global_step,
                                                               write_meta_graph=False)
        self.log.info('Checkpoint saved to {}'.format(saved_checkpoint_path))
        return saved_checkpoint_path

    def load_test_checkpoint(self, save_path):
        reader = tf.train.NewCheckpointReader(save_path)
        saved_shapes = reader.get_variable_to_shape_map()
        self.log.info('Saved Vars:\n' + str(saved_shapes.keys()))
        for name in saved_shapes.keys():
            self.log.info(
                'Name: {}, Tensor: {}'.format(name, reader.get_tensor(name)))

    def setup_model(self, weights_name='Weights', bias_name='Bias'):
        """Set up simple tensorflow model."""
        tf.reset_default_graph()

        self.global_step = tf.get_variable(
                'global_step', [],
                dtype=tf.int64, trainable=False,
                initializer=tf.constant_initializer(0))

        # Model parameters and placeholders.
        x = tf.placeholder(tf.float32, name='x')
        y = tf.placeholder(tf.float32, name='y')
        W = tf.get_variable(weights_name, [1], dtype=tf.float32)
        b = tf.get_variable(bias_name, [1], dtype=tf.float32)

        # Model output, loss and optimizer.
        linear_model = W * x + b
        loss = tf.reduce_sum(tf.square(linear_model - y))
        optimizer_base = tf.train.GradientDescentOptimizer(0.01)

        # Model train op.
        optimizer = optimizer_base.minimize(
            loss, global_step=self.global_step)

        # Train targets.
        self.train_targets = {'loss': loss,
                              'optimizer': optimizer}

    @classmethod
    def setup_log(cls):
        cls.log = logging.getLogger(':'.join([__name__, cls.__name__]))
        cls.log.setLevel('DEBUG')

    @classmethod
    def setup_conn(cls):
        cls.conn = pymongo.MongoClient(host=cls.HOST, port=cls.PORT)

    @classmethod
    def setup_cache(cls):
        cls.cache_dir = os.path.join(cls.CACHE_DIR,
                                     '%s:%d' % (cls.HOST, cls.PORT),
                                     cls.DATABASE_NAME,
                                     cls.COLLECTION_NAME,
                                     cls.EXP_ID)
        cls.makedirs(cls.cache_dir)
        cls.save_path = os.path.join(cls.cache_dir, 'checkpoint')

    @classmethod
    def setup_params(cls):
        cls.model_params = {'func': model.mnist_tfutils_new,
                            'devices': ['/gpu:0', '/gpu:1'],
                            'prefix': 'model_0'}

        cls.save_params = {
            'host': cls.HOST,
            'port': cls.PORT,
            'dbname': cls.DATABASE_NAME,
            'collname': cls.COLLECTION_NAME,
            'exp_id': cls.EXP_ID,
            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        cls.train_params = {
                'data_params': {'func': data.build_data,
                    'batch_size': 100,
                    'group': 'train',
                    'directory': TFUTILS_HOME},
                'num_steps': 500}

        cls.loss_params = {
                'targets': ['labels'],
                'agg_func': tf.reduce_mean,
                'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits}

        cls.load_params = {'do_restore': True}

        cls.optimizer_params = {'func': optimizer.ClipOptimizer,
                                'optimizer_class': tf.train.MomentumOptimizer,
                                'clip': True,
                                'momentum': 0.9}

        cls.learning_rate_params = {'learning_rate': 0.05,
                                    'decay_steps': 10000 // 256,
                                    'decay_rate': 0.95,
                                    'staircase': True}
        cls.params = {
            'dont_run': False,
            'skip_check': True,
            'model_params': cls.model_params,
            'train_params': cls.train_params,
            'validation_params': {},
            'log_device_placement': False,
            'save_params': cls.save_params,
            'load_params': cls.load_params,
            'loss_params': cls.loss_params,
            'optimizer_params': cls.optimizer_params,
            'learning_rate_params': cls.learning_rate_params}

    @classmethod
    def remove_checkpoint(cls, checkpoint):
        """Remove a tf.train.Saver checkpoint."""
        cls.log.info('Removing checkpoint: {}'.format(checkpoint))
        # TODO: remove ckpt
        cls.log.info('Checkpoint successfully removed.')
        raise NotImplementedError

    @classmethod
    def remove_directory(cls, directory):
        """Remove a directory."""
        cls.log.info('Removing directory: {}'.format(directory))
        shutil.rmtree(directory)
        cls.log.info('Directory successfully removed.')

    @classmethod
    def remove_database(cls, database_name):
        """Remove a MonogoDB database."""
        cls.log.info('Removing database: {}'.format(database_name))
        cls.conn.drop_database(database_name)
        cls.log.info('Database successfully removed.')

    @classmethod
    def remove_collection(cls, collection_name):
        """Remove a MonogoDB collection."""
        cls.log.debug('Removing collection: {}'.format(collection_name))
        cls.conn[cls.DATABASE_NAME][collection_name].drop()
        cls.log.info('Collection successfully removed.')

    @classmethod
    def remove_document(cls, document):
        raise NotImplementedError

    @staticmethod
    def makedirs(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == '__main__':
    unittest.main()
