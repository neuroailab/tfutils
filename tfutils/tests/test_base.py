"""Test base module."""

import os
import sys
import errno
import shutil
import cPickle
import logging
import unittest

import gridfs
import pymongo
import tensorflow as tf

sys.path.insert(0, "..")

import base
import tfutils.data as data
import tfutils.model as model
import tfutils.utils as utils


num_batches_per_epoch = 10000 // 256


def setUpModule():
    """Set up module once, before any TestCases are run."""
    logging.basicConfig()


def tearDownModule():
    """Tear down module after all TestCases are run."""
    pass


class TestBase(unittest.TestCase):

    port = 29101
    host = 'localhost'
    database_name = 'TFUTILS_TEST_DB'

    @classmethod
    def setUpClass(cls):
        """Set up class once before any test methods are run."""
        cls.log = logging.getLogger(':'.join([__name__, cls.__name__]))
        cls.log.setLevel('DEBUG')

        # Open primary MongoDB connection.
        cls.conn = pymongo.MongoClient(host=cls.host,
                                       port=cls.port)

        cls.collection_name = cls.__name__
        cls.collection = cls.conn[cls.database_name][cls.collection_name]

    @classmethod
    def tearDownClass(cls):
        """Tear down class after all test methods have run."""
        # TODO: Remove cache.
        cls.remove_database(cls.database_name)
        [cls.conn.drop_database(x)
         for x in cls.conn.database_names()
         if x.startswith(cls.database_name)]

        # Close primary MongoDB connection.
        cls.conn.close()

    def setUp(self):
        """Set up class before each test method is executed."""
        pass

    def tearDown(self):
        """Tear Down is called after each test method is executed."""
        self.remove_collection(self.collection_name)

    def setup_params(self, exp_id):

        params = {}
        params['model_params'] = {
            'func': model.mnist_tfutils}

        params['save_params'] = {
            'host': self.host,
            'port': self.port,
            'dbname': self.database_name,
            'collname': self.collection_name,
            'exp_id': exp_id,
            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        params['train_params'] = {
            'data_params': {'func': data.MNIST,
                            'batch_size': 100,
                            'group': 'train',
                            'n_threads': 4},
            'queue_params': {'queue_type': 'fifo',
                             'batch_size': 100},
            'num_steps': 500}

        params['learning_rate_params'] = {
            'learning_rate': 0.05,
            'decay_steps': num_batches_per_epoch,
            'decay_rate': 0.95,
            'staircase': True}

        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    'func': data.MNIST,
                    'batch_size': 100,
                    'group': 'test',
                    'n_threads': 4},
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': 100},
                'num_steps': 10,
                'agg_func': utils.mean_dict}}

        params['skip_check'] = True

        return params

    def test_training(self):
        """Illustrate training.

        This test illustrates how basic training is performed using the
        tfutils.base.train_from_params function.  This is the first in a sequence of
        interconnected tests. It creates a pretrained model that is used by
        the next few tests (test_validation and test_feature_extraction).

        As can be seen by looking at how the test checks for correctness, after the
        training is run, results of training, including (intermittently) the full
        variables needed to re-initialize the tensorflow model, are stored in a
        MongoDB.

        Also see docstring of the tfutils.base.train_from_params function for more detailed
        information about usage.

        """
        exp_id = 'training0'
        params = self.setup_params(exp_id)

        # Run training.
        base.train_from_params(**params)

        # Test if results are as expected.
        self.assert_as_expected(exp_id, count=26, step=[0, 200, 400])
        r = self.collection['files'].find({'exp_id': exp_id, 'step': 0})[0]
        self.asserts_for_record(r, params, train=True)
        r = self.collection['files'].find({'exp_id': exp_id, 'step': 20})[0]
        self.asserts_for_record(r, params, train=True)

        # Run another 500 steps of training on the same experiment id.
        params['train_params']['num_steps'] = 1000
        base.train_from_params(**params)

        # Test if results are as expected.
        self.assert_as_expected(exp_id, 51, [0, 200, 400, 600, 800, 1000])
        self.assertEqual(self.collection['files'].distinct('exp_id'), [exp_id])

        r = self.collection['files'].find({'exp_id': exp_id, 'step': 1000})[0]
        self.asserts_for_record(r, params, train=True)

        # Run 500 more steps but save to a new experiment id.
        new_exp_id = 'training1'
        params['train_params']['num_steps'] = 1500
        params['load_params'] = {'exp_id': exp_id}
        params['save_params']['exp_id'] = new_exp_id

        base.train_from_params(**params)

        self.assert_step(new_exp_id, [1200, 1400])

    def test_interactive_training(self):
        pass

    def test_training_save(self):
        """Illustrate saving to the grid file system during training time."""
        exp_id = 'training_save'
        params = self.setup_params(exp_id)

        params['save_params']['save_to_gfs'] = ['first_image']
        params['save_params']['save_valid_freq'] = 3000
        params['save_params']['save_filters_freq'] = 30000
        params['save_params']['cache_filters_freq'] = 3000
        params['train_params']['targets'] = {'func': self.get_first_image_target}

        # Actually run the training.
        base.train_from_params(**params)

        # Check that the first image has been saved.
        coll = self.collection['files']
        q = {'exp_id': exp_id, 'train_results': {'$exists': True}}
        train_steps = coll.find(q)
        self.assertEqual(train_steps.count(), 5)
        idx = train_steps[0]['_id']
        fn = coll.find({'item_for': idx})[0]['filename']
        fs = gridfs.GridFS(coll.database, self.collection_name)
        fh = fs.get_last_version(fn)
        saved_data = cPickle.loads(fh.read())
        fh.close()

        self.assertIn('train_results', saved_data)
        self.assertIn('first_image', saved_data['train_results'])
        self.assertEqual(len(saved_data['train_results']['first_image']), 100)
        self.assertEqual(saved_data['train_results']['first_image'][0].shape, (28 * 28,))

    def test_validation(self):
        """Illustrate validation.

        This is a test illustrating how to compute performance on a trained model on a new dataset,
        using the tfutils.base.test_from_params function.  This test assumes that test_training function
        has run first (to provide a pre-trained model to validate).

        After the test is run, results from the validation are stored in the MongoDB.
        (The test shows how the record can be loaded for inspection.)

        See the docstring of tfutils.base.test_from_params for more detailed information on usage.

        """
        # Specify the parameters for the validation.
        exp_id = 'training0'
        val_exp_id = 'validation0'

        params = self.setup_params(exp_id)

        params.pop('train_params')
        params.pop('learning_rate_params')
        params['load_params'] = params['save_params']
        params['save_params'] = {'exp_id': val_exp_id}

        # Actually run the model.
        base.test_from_params(**params)

        # ... specifically, there is now a record containing the validation0 performance results
        self.assertEqual(self.collection['files'].find({'exp_id': val_exp_id}).count(), 1)
        # ... here's how to load the record:
        r = self.collection['files'].find({'exp_id': val_exp_id})[0]
        self.asserts_for_record(r, params, train=False)

        # ... check that the recorrectly ties to the id information for the
        # pre-trained model it was supposed to validate
        self.assertTrue(r['validates'])
        idval = self.collection['files'].find({'exp_id': exp_id})[50]['_id']
        v = self.collection['files'].find({'exp_id': val_exp_id})[0]['validates']
        self.assertEqual(idval, v)

    def assert_count(self, exp_id, count):
        self.assertEqual(
            self.collection['files'].find({'exp_id': exp_id}).count(),
            count)

    def assert_step(self, exp_id, step):
        self.assertEqual(
            self.collection['files']
                .find({'exp_id': exp_id, 'saved_filters': True})
                .distinct('step'),
            step)

    def assert_as_expected(self, exp_id, count=None, step=None):
        if count is not None:
            self.assert_count(exp_id, count)
        if step is not None:
            self.assert_step(exp_id, step)

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
        cls.log.info('Removing collection: {}'.format(collection_name))
        cls.conn[cls.database_name][collection_name].drop()
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

    @staticmethod
    def get_first_image_target(inputs, outputs, **ttarg_params):
        """Return target for saving the first image of every batch.

        Used in test_training_save test to test save_to_gfs option.

        """
        return {'first_image': inputs['images'][0]}

    @staticmethod
    def asserts_for_record(r, params, train=False):
        if r.get('saved_filters'):
            assert r['_saver_write_version'] == 2
            assert r['_saver_num_data_files'] == 1
        assert type(r['duration']) == float

        should_contain = ['save_params', 'load_params',
                          'model_params', 'validation_params']
        assert set(should_contain).difference(r['params'].keys()) == set()

        vk = r['params']['validation_params'].keys()
        vk1 = r['validation_results'].keys()
        assert set(vk) == set(vk1)

        assert r['params']['model_params']['seed'] == 0
        print(r['params']['model_params']['func']['modname'])
        assert r['params']['model_params']['func']['modname'] == 'tfutils.model'
        assert r['params']['model_params']['func']['objname'] == 'mnist_tfutils'
        assert set(['hidden1', 'hidden2', u'softmax_linear']).difference(
            r['params']['model_params']['cfg_final'].keys()) == set()

        _k = vk[0]
        should_contain = ['agg_func', 'data_params', 'num_steps',
                          'online_agg_func', 'queue_params', 'targets']
        assert set(should_contain).difference(
            r['params']['validation_params'][_k].keys()) == set()

        if train:
            assert r['params']['model_params']['train'] is True
            for k in ['num_steps', 'queue_params']:
                assert r['params']['train_params'][k] == params['train_params'][k]

            should_contain = ['loss_params', 'optimizer_params',
                              'train_params', 'learning_rate_params']
            assert set(should_contain).difference(r['params'].keys()) == set()
            assert r['params']['train_params']['thres_loss'] == 100
            assert r['params']['train_params']['data_params']['func']['modname'] == 'tfutils.data'
            assert r['params']['train_params']['data_params']['func']['objname'] == 'MNIST'

            assert r['params']['loss_params']['agg_func']['modname'] == 'tensorflow.python.ops.math_ops'
            assert r['params']['loss_params']['agg_func']['objname'] == 'reduce_mean'
            assert r['params']['loss_params']['loss_per_case_func']['modname'] == 'tensorflow.python.ops.nn_ops'
            assert r['params']['loss_params']['loss_per_case_func']['objname'] == 'sparse_softmax_cross_entropy_with_logits'
            assert r['params']['loss_params']['targets'] == ['labels']
        else:
            assert not r['params']['model_params']['train']
            assert 'train_params' not in r['params']


class TestDistributedModel(TestBase):

    def setup_params(self, exp_id):

        params = {}
        params['model_params'] = {
            'func': model.mnist_tfutils,
            'devices': ['/gpu:0', '/gpu:1']}

        params['save_params'] = {
            'host': self.host,
            'port': self.port,
            'dbname': self.database_name,
            'collname': self.collection_name,
            'exp_id': exp_id,
            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        params['train_params'] = {
            'data_params': {
                'func': data.MNIST,
                'batch_size': 100,
                'group': 'train',
                'n_threads': 4},
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': 100},
            'num_steps': 500}

        params['learning_rate_params'] = {
            'learning_rate': 0.05,
            'decay_steps': num_batches_per_epoch,
            'decay_rate': 0.95,
            'staircase': True}

        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    'func': data.MNIST,
                    'batch_size': 100,
                    'group': 'test',
                    'n_threads': 4},
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': 100},
                'num_steps': 10,
                'agg_func': utils.mean_dict}}

        params['skip_check'] = True

        return params


@unittest.skip('skipping')
class TestMultiModel(TestBase):

    def setup_params(self, exp_id):

        params = {}
        params['model_params'] = [
            {'func': model.mnist_tfutils},
            {'func': model.mnist_tfutils}]

        params['save_params'] = {
            'host': self.host,
            'port': self.port,
            'dbname': self.database_name,
            'collname': self.collection_name,
            'exp_id': exp_id,
            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        params['train_params'] = {
            'data_params': {'func': data.MNIST,
                            'batch_size': 100,
                            'group': 'train',
                            'n_threads': 4},
            'queue_params': {'queue_type': 'fifo',
                             'batch_size': 100},
            'num_steps': 500}

        params['learning_rate_params'] = {
            'learning_rate': 0.05,
            'decay_steps': num_batches_per_epoch,
            'decay_rate': 0.95,
            'staircase': True}

        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    'func': data.MNIST,
                    'batch_size': 100,
                    'group': 'test',
                    'n_threads': 4},
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': 100},
                'num_steps': 10,
                'agg_func': utils.mean_dict}}

        params['skip_check'] = True

        def test_training(self):
            base_exp_id = 'training0'
            params = self.setup_params(base_exp_id)
            for i in range(num_models):
                exp_id = base_exp_id + '_model_{}'.format(i)

        return params


@unittest.skip('skipping')
class TestDistributedMultiModel(TestMultiModel):

    def setup_params(self, exp_id):

        params = {}
        params['model_params'] = [
            {'func': model.mnist_tfutils,
             'devices': ['/gpu:0', '/gpu:1']},
            {'func': model.mnist_tfutils,
             'devices': ['/gpu:2', '/gpu:3']}]

        params['save_params'] = {
            'host': self.host,
            'port': self.port,
            'dbname': self.database_name,
            'collname': self.collection_name,
            'exp_id': exp_id,
            'save_valid_freq': 20,
            'save_filters_freq': 200,
            'cache_filters_freq': 100}

        params['train_params'] = {
            'data_params': {'func': data.MNIST,
                            'batch_size': 100,
                            'group': 'train',
                            'n_threads': 4},
            'queue_params': {'queue_type': 'fifo',
                             'batch_size': 100},
            'num_steps': 500}

        params['learning_rate_params'] = {
            'learning_rate': 0.05,
            'decay_steps': num_batches_per_epoch,
            'decay_rate': 0.95,
            'staircase': True}

        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    'func': data.MNIST,
                    'batch_size': 100,
                    'group': 'test',
                    'n_threads': 4},
                'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': 100},
                'num_steps': 10,
                'agg_func': utils.mean_dict}}

        params['skip_check'] = True

        return params


if __name__ == '__main__':
    unittest.main()
