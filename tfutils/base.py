from __future__ import absolute_import, division, print_function

import sys, math, time, threading
import tensorflow as tf


class CustomQueue(object):

    def __init__(self, data, batch_size=128):
        """
        A generic queue for reading data

        Based on https://indico.io/blog/tensorflow-data-input-part2-extensions/
        """
        self.data = data
        self.batch_size = batch_size

        dtypes = [d.dtype for d in data.data_node.values()]
        shapes = [d.get_shape() for d in data.data_node.values()]
        self.queue = tf.RandomShuffleQueue(capacity=1024 * 16 + 4 * batch_size,
                                           min_after_dequeue=1024 * 16,
                                           dtypes=dtypes,
                                           shapes=shapes)
        self.enqueue_op = self.queue.enqueue(data.data_node.values())
        data_batch = self.queue.dequeue_many(batch_size)
        self.data_batch = {k:v for k,v in zip(data.data_node.keys(), data_batch)}

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for batch in self.data:
            sess.run(self.enqueue_op, feed_dict=batch)

    def start_threads(self, sess, n_threads=4):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


class Saver(tf.train.Saver):

    def __init__(self,
                 sess,
                 restore_vars=False,
                 restore_var_file='',
                 save_vars=True,
                 save_vars_freq=3000,
                 save_path='',
                 save_loss=True,
                 save_loss_freq=5,
                 *args, **kwargs):
        """
        Output printer, logger and saver to a database

        NOTE: Not working yet
        """

        super(Saver, self).__init__(*args, **kwargs)
        self.sess = sess
        self.save_vars = save_vars
        self.save_vars_freq = save_vars_freq
        self.save_path = save_path
        self.save_loss = save_loss
        self.save_loss_freq = save_loss_freq
        if restore_vars:
            self.restore(self.sess, restore_var_file)
            print('Variables restored')

        self.start_time_step = time.time()  # start timer

    def save(self, step, results):
        elapsed_time_step = time.time() - start_time_step
        start_time_step = time.time()

        if math.isnan(results['loss']):
            raise Exception('Model diverged with loss = NaN')

        if self.save_vars and step % self.save_vars_freq == 0 and step > 0:
            super(Saver, self).save(sess,
                                    save_path=self.save_path,
                                    global_step=step,
                                    write_meta_graph=False)
            import pdb; pdb.set_trace()  # check if global_step == step
            print('saved variable checkpoint')

        if step > 0:
            # write loss to file
            if self.save_loss and step % self.save_loss_freq == 0:

                # tot_losses[step // self.save_loss_freq] %
                #     tot_losses_len - 1, :] = [step, results['tot_loss']]
                pass

        print('Step {} -- total_loss: {:.6f}, lr: {:.6f}, time: {:.1f} ms'.format(
              step, results['tot_loss'], results['lr'], 1000 * elapsed_time_step))

        if step % write_out_freq == 0:
            # Write to file. only every EVAL_FREQUENCY to limit I/O
            if params['save_loss']:
                with open(outfile_loss, 'ab') as f_handle:
                    np.savetxt(
                        f_handle,  # file name
                        tot_losses,  # array to save
                        fmt='%.3f',  # formatting, 3 digits
                        delimiter=',',  # column delimiter
                        newline='\n')  # new line character
        sys.stdout.flush()  # flush the stdout buffer


def run(params, queue, logits, targets, start_step, end_step):
    """
    Args:
        - params: dictionary of params
        - queue
        - logits
        - targets
    """
    # Create a variable to count the number of train() calls.
    # This equals the number of batches processed.
    # TODO: do we actually need it?
    # global_step = tf.get_variable('global_step', [],
    #                               initializer=tf.constant_initializer(0),
    #                               trainable=False)

    # create session
    sess = tf.Session(config=tf.ConfigProto(
                      allow_soft_placement=True,
                      log_device_placement=params['log_device_placement']))

    if params['tensorboard']:  # save graph to tensorboard
        tf.train.SummaryWriter(params['tensorboard_dir'], sess.graph)

    # initialize and/or restore variables for graph
    init = tf.initialize_all_variables()
    sess.run(init)
    print('variables initialized')

    tf.train.start_queue_runners(sess=sess)
    # start our custom queue runner's threads
    queue.start_threads(sess, n_threads=params['num_preprocess_threads'])

    saver = Saver(sess,
                  restore_vars=params['restore_vars'],
                  restore_var_file=params['restore_var_file'],
                  save_vars=params['save_vars'],
                  save_vars_freq=params['save_vars_freq'],
                  save_path=params['save_path'],
                  save_loss=params['save_loss'],
                  save_loss_freq=params['save_loss_freq'],
                  max_to_keep=params['max_to_keep'])
    # start_time_step = time.time()  # start timer
    for step in xrange(start_step, end_step):
        # get run output as dictionary {'2': loss2, 'lr': lr, etc..}
        results = sess.run(targets)
        # print output, save variables to checkpoint and save loss etc
        # saver.save(step, results)
    sess.close()
