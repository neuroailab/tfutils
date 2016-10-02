from __future__ import absolute_import, division, print_function

import sys, time

import numpy as np
import pymongo
import tensorflow as tf
from tensorflow.contrib.learn import NanLossDuringTrainingError


class Saver(tf.train.Saver):

    def __init__(self,
                 sess,
                 dbname=None,
                 collname='sim',
                 exp_id=None,
                 port=31001,
                 restore_vars=False,
                 restore_var_file='',
                 start_step=None,
                 save_vars=True,
                 save_vars_freq=3000,
                 save_path='',
                 save_loss=True,
                 save_loss_freq=5,
                 tensorboard=False,
                 tensorboard_dir='',
                 *args, **kwargs):
        """
        Output printer, logger and saver to a database

        NOTE: Not working yet
        """
        super(Saver, self).__init__(*args, **kwargs)
        if dbname is not None:
            self.conn = pymongo.MongoClient(port=port)
            self.coll = self.conn[dbname][collname]
        else:
            # TODO: save locally
            raise ValueError('Please specify database name for storing data')
        self.sess = sess
        self.exp_id = exp_id
        self.save_vars = save_vars
        self.save_vars_freq = save_vars_freq
        self.save_path = save_path
        self.save_loss = save_loss
        self.save_loss_freq = save_loss_freq
        if restore_vars:
            self.restore(self.sess, restore_var_file)
            print('Variables restored')

        if tensorboard:  # save graph to tensorboard
            tf.train.SummaryWriter(tensorboard_dir, tf.get_default_graph())

        self.start_time_step = time.time()  # start timer

    def save(self, step, results):
        elapsed_time_step = time.time() - self.start_time_step
        self.start_time_step = time.time()

        if np.isnan(results['loss']):
            raise NanLossDuringTrainingError

        if self.save_vars and step % self.save_vars_freq == 0 and step > 0:
            super(Saver, self).save(self.sess,
                                    save_path=self.save_path,
                                    global_step=step,
                                    write_meta_graph=False)
            print('saved variable checkpoint')

        rec = {'exp_id': self.exp_id,
            #    'cfg': preprocess_config(cfg),
            #    'saved_filters': saved_filters,
                'kind': 'train',
                'step': step,
                'loss': results['loss'],
                'learning_rate': results['lr'],
                'duration': 1000 * elapsed_time_step}

        if step > 0:
            # write loss to file
            if self.save_loss and step % self.save_loss_freq == 0:
                pass
                # self.coll.insert(rec)

        print('Step {} -- loss: {:.6f}, lr: {:.6f}, time: {:.0f}'
              'ms'.format(rec['step'], rec['loss'], rec['learning_rate'], rec['duration']))
        sys.stdout.flush()  # flush the stdout buffer

    def valid(self, step, results):
        elapsed_time_step = time.time() - self.start_time_step
        self.start_time_step = time.time()
        rec = {'exp_id': self.exp_id,
            #    'cfg': preprocess_config(cfg),
            #    'saved_filters': saved_filters,
               'kind': 'validation',
               'step': step,
               'duration': 1000 * elapsed_time_step}
        rec.update(results)
        if step > 0:
            # write loss to file
            if self.save_valid and step % self.save_valid_freq == 0:
                pass
                # self.coll.insert(rec)

        message = ('Step {} validation -- ' +
                   '{}: {:.3f}, ' * len(results) +
                   '{:.0f} ms')
        args = []
        for k, v in results.items():
            args.extend([k,v])
        args = [rec['step']] + args + [rec['duration']]
        print(message.format(*args))

    def predict(self, step, results):
        if not hasattr(results['output'], '__iter__'):
            outputs = [results['outputs']]
        else:
            outputs = results['outputs']

        preds = [tf.argmax(output, 1) for output in outputs]

        return preds

    def test(self, step, results):
        raise NotImplementedError


def run(sess, queues, saver, train_targets, valid_targets=None,
        start_step=0, end_step=None):
    """
    Args:
        - queues (~ data)
        - saver
        - targets
    """
    # initialize and/or restore variables for graph
    init = tf.initialize_all_variables()
    sess.run(init)
    print('variables initialized')

    tf.train.start_queue_runners(sess=sess)
    # start our custom queue runner's threads
    if not hasattr(queues, '__iter__'):
        queues = [queues]
    for queue in queues:
        queue.start_threads(sess)

    # start_time_step = time.time()  # start timer
    print('start training')
    for step in xrange(start_step, end_step):
        # get run output as dictionary {'2': loss2, 'lr': lr, etc..}
        results = sess.run(train_targets)
        # print output, save variables to checkpoint and save loss etc
        saver.save(step, results)
    sess.close()
