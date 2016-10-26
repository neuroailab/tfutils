from __future__ import absolute_import, division, print_function

import os, sys, time, math, importlib, argparse, json

import numpy as np
import pymongo
import tensorflow as tf
import gridfs

import tfutils.error as error
import tfutils.data as data


class Saver(tf.train.Saver):

    def __init__(self,
                 sess,
                 params,
                 host='localhost',
                 port=31001,
                 dbname='testdb',
                 collname='testcoll',
                 exp_id='test',
                 save=True,
                 restore=True,
                 save_metrics_freq=5,
                 save_valid_freq=3000,
                 cache_filters_freq=3000,
                 cache_path=None,
                 save_filters_freq=30000,
                 tensorboard_dir=None,
                 start_step=0,
                 end_step=None,
                 force_fetch=False,
                 *args, **kwargs):
        """
        Output printer, logger and saver to a database
        Kwargs:
        - save_vars_freq
            0 or None not to save. Otherwise the number of steps.
        - restore_vars_file
            If None, don't save
        NOTE: Not working yet
        """
        super(Saver, self).__init__(*args, **kwargs)
        self.sess = sess
        self.params = params
        self.exp_id = exp_id
        self.dosave = save
        self.save_metrics_freq = save_metrics_freq
        self.save_valid_freq = save_valid_freq
        self.cache_filters_freq = cache_filters_freq
        self.save_filters_freq = save_filters_freq

        if self.dosave:
            self.conn = pymongo.MongoClient(host=host, port=port)
            self.coll = self.conn[dbname][collname + '.files']
            self.collfs = gridfs.GridFS(self.conn[dbname], collname)
            recent_name = '_'.join([dbname, collname, exp_id, '__RECENT'])
            self.collfs_recent = gridfs.GridFS(self.conn[recent_name])

        if restore:
            self.restore_model(force_fetch=force_fetch)
            print('Variables restored')

        if tensorboard_dir is not None:  # save graph to tensorboard
            tf.train.SummaryWriter(tensorboard_dir, tf.get_default_graph())

        if cache_path is None:
            self.cache_path = os.path.join(os.environ['HOME'], '.tfutils')
        else:
            self.cache_path = cache_path
        # import pdb; pdb.set_trace()
        if not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)

        self.start_time_step = time.time()  # start timer

    def restore_model(self, force_fetch=False):
        """
        Fetches record, saves locally, then uses tf's saver.restore
        """
        # fetch record from database and get the filename info from record
        rec, cache_filename = self.load_from_db({'exp_id': self.exp_id, 
        									     'saved_filters': True},
        									     cache_model=True)
        # tensorflow restore
        self.restore(self.sess, cache_filename)
        print('Model variables restored.')

    def load_from_db(self, query, cache_model=False):
        """
        Loads checkpoint from the database
        Checks the recent and regular checkpoint fs to find the latest one
        matching the query. Returns the GridOut obj corresponding to the
        record.
        Args:
            query: dict of Mongo queries
        """
        count = self.collfs.find(query).count()
        if count > 0:  # get latest that matches query
            ckpt_record = self.collfs._GridFS__files.find(query,
                            sort=[('uploadDate', -1)])[0]
            loading_from = 'long-term storage'

        count_recent = self.collfs_recents.find(query).count()
        if count_recent > 0:  # get latest that matches query
            ckpt_record_rec = self.collfs_recent._GridFS__files.find(query,
                                sort=[('uploadDate', -1)])[0]
            # use the record with latest timestamp
            if ckpt_record is None or ckpt_record_rec['uploadDate'] > ckpt_record['uploadDate']:
                loading_from = 'recent storage'
                ckpt_record = ckpt_record_rec

        if count + count_recent == 0:  # no matches for query
            raise Exception('No matching checkpoint for query "{}"'.format(repr(query)))

        print('Loading checkpoint from ', loading_from)
        
        if cache_model:
			# should be of form *-1000 (step)
			filename = os.path.basename(ckpt_record.filename.split)
			cache_filename = os.path.join(self.cache_path, filename)
			# TODO: check that these filenames are unique

			# check if there is no local copy
			if not os.path.isfile(cache_filename) and not force_fetch:
				# create new file to write from gridfs
				load_dest = open(cache_filename, "w+")
				load_dest.close()
				load_dest = open(cache_filename, 'rwb+')
				fsbucket = gridfs.GridFSBucket(ckpt_record._GridOut__files.database,
								bucket_name=ckpt_record._GridOut__files.name.split('.')[0])
				# save to local disk
				fsbucket.download_to_stream(ckpt_record._id, load_dest)        
		else:
			cache_filename = None
        
        return ckpt_record, cache_filename

    def save(self, step, train_res, valid_res=None):
        elapsed_time_step = time.time() - self.start_time_step
        just_saved = False  # for saving filters

        rec = {'exp_id': self.exp_id,
               'params': self.params,
               'saved_filters': False,
               'step': step,
               'duration': int(1000 * elapsed_time_step)}

        # print loss, learning rate etc
        if self.save_metrics_freq not in (None, False, 0):
            if step % self.save_metrics_freq == 0:
                for k, v in train_res.items():
                    if isinstance(v, np.float):
                        train_res[k] = float(v)
                rec.update(train_res)
                del rec['opt']
                # TODO: also include error rate of the train set to monitor overfitting

                message = 'Step {} ({:.0f} ms) -- '.format(rec['step'], rec['duration'])
                message += ', '.join(['{}: {:.4f}'.format(k,v) for k,v in train_res.items() if k != 'opt'])
                print(message)

        # print validation set performance etc
        if self.save_valid_freq not in (None, False, 0) and valid_res is not None:
            if step % self.save_valid_freq == 0:
                for k, v in valid_res.items():
                    if isinstance(v, np.float):
                        valid_res[k] = float(v)
                rec.update(valid_res)

                message = 'Step {} ({:.0f} ms) validation -- '.format(rec['step'], rec['duration'])
                message += ', '.join('{}: {}'.format(k,v) for k,v in valid_res.items())
                print(message)

        # save to db -- ERROR HERE? REMOVE THIS?
        if self.dosave:
            self.coll.insert_one(rec)

        # save filters to cache and recent db
        if self.cache_filters_freq not in (None, False, 0) and self.dosave:
            if step % self.cache_filters_freq == 0 and step > 0:
                saved_path = super(Saver, self).save(self.sess,
                                                     save_path=self.cache_path,
                                                     global_step=step,
                                                     write_meta_graph=False)
                rec['saved_filters'] = True
                self.collfs_recent.put(open(saved_path, 'rb'),
                                       filename=saved_path,
                                       **rec)
                just_saved = True
                print('Saved filters to the recent database.')

        # save filters to db
        if self.save_filters_freq not in (None, False, 0):
            if step % self.save_filters_freq == 0 and step > 0:
                if not just_saved:  # avoid resaving if already done above
                    saved_path = super(Saver, self).save(self.sess,
                                                      save_path=self.cache_path,
                                                      global_step=step,
                                                      write_meta_graph=False)
                rec['saved_filters'] = True
                self.collfs_recent.put(open(saved_path, 'rb'),
                                       filename=saved_path,
                                       **rec)
                print('Saved filters to the long-term database.')

        sys.stdout.flush()  # flush the stdout buffer
        self.start_time_step = time.time()


def predict(step, results):
    if not hasattr(results['output'], '__iter__'):
        outputs = [results['outputs']]
    else:
        outputs = results['outputs']

    preds = [tf.argmax(output, 1) for output in outputs]

    return preds


def test(step, results):
    raise NotImplementedError


def run(sess, queues, saver, train_targets, valid_targets=None,
        start_step=0, end_step=None, thres_loss=100):
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
        train_results = sess.run(train_targets)
        if train_results['loss'] > thres_loss:
            raise error.HiLossError('Loss {:.2f} exceeded the threshold {:.2f}'.format(train_results['loss'], thres_loss))

        if valid_targets is not None:
            valid_results = sess.run(valid_targets)
        else:
            valid_results = None
        # print output, save variables to checkpoint and save loss etc
        saver.save(step, train_results, valid_results)
    sess.close()


def run_base(model_func,
             model_kwargs,
             train_data_func,
             train_data_kwargs,
             loss_func,
             loss_kwargs,
             lr_func,
             lr_kwargs,
             opt_func,
             opt_kwargs,
             saver_kwargs,
             train_targets_func=None,
             train_targets_kwargs={},
             valid_data_func=None,
             valid_data_kwargs={},
             valid_targets_func=None,
             valid_targets_kwargs=None,
             thres_loss=100,
             seed=None,
             start_step=0,
             end_step=float('inf'),
             log_device_placement=True
             ):

    with tf.Graph().as_default():  # to have multiple graphs [ex: eval, train]
        tf.get_variable('global_step', [],
                        initializer=tf.constant_initializer(0),
                        trainable=False)

		#train_data_func returns dictionary of iterators, with one key per input to model
		n_threads = train_data_kwargs.pop('n_threads')
        train_inputs = train_data_func(**train_data_kwargs)
        queues = [CustomQueue(train_inputs.node, 
        					  train_inputs, 
        					  queue_batch_size=train_provider.batch_size, 
        					  n_threads=n_threads)]
        
		assert 'cfg0' in model_kwargs
		assert 'seed' in model_kwargs
        train_outputs, cfg1 = model_func(inputs=train_inputs, 
              			                train=True, 
        			                    **model_kwargs)

        loss = loss_func(train_inputs, train_outputs, **loss_kwargs)
        lr = lr_func(**lr_kwargs)
        optimizer = opt_func(loss, lr, **opt_kwargs)

        train_targets = {'loss': loss, 'lr': lr, 'opt': optimizer}

        if train_targets_func is not None:
            ttarg = train_targets_func(train_inputs, train_outputs, **train_targets_kwargs)
            train_targets.update(ttarg)

        if valid_data_func is not None:
            valid_inputs = valid_data_func(**valid_data_kwargs)
            new_queue = CustomQueue(valid_inputs.node, 
        					 	    valid_inputs, 
        					  	    queue_batch_size=valid_provider.batch_size, 
        					  	    n_threads=n_threads)
            
            queues += [new_queue]       
            new_model_kwargs = copy.deepcopy(model_kwargs)
            new_model_kwargs['seed'] = None
            new_model_kwargs['cfg0'] = cfg     
            valid_outputs, _cfg = model_func(inputs=valid_inputs, 
            						         train=False, 
            						         **new_model_kwargs)
            assert cfg1 == _cfg, (cfg1, _cfg)
            valid_targets = valid_targets_func(valid_inputs,
            						          valid_outputs, 
            						          **valid_targets_kwargs)
        else:
            valid_targets = None

        # create session
        sess = tf.Session(config=tf.ConfigProto(
                                allow_soft_placement=True,
                                log_device_placement=log_device_placement))

		model_kwargs_final = copy.deepcopy(model_kwargs)
		model_kwargs_final['cfg1'] = cfg1
	    params = {'model_kwargs': model_kwargs_final,
	    	      'train_data_kwargs': train_data_kwargs,
	    	      'loss_kwargs': loss_kwargs,
	    	      'optimizer_kwargs': opt_kwargs,
	    	      'valid_targets_kwargs': valid_targets_kwargs
	    	      #... other stuff?  how are the function names passed? 
	    	      } 
	    	      
        saver = Saver(sess, params, **saver_kwargs)
        run(sess,
            queues,
            saver,
            train_targets=train_targets,
            valid_targets=valid_targets,
            start_step=start_step,
            end_step=end_step,
            thres_loss=thres_loss)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', default=None)
    parser.add_argument('-g', '--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.params is not None:
        params = json.load(open(args.params))
        return params
    else:
        return None
