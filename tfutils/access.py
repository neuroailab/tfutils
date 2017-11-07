'''
Tools to easily and efficiently load data from the tfutils mongo database,
'''
import pymongo as pm
import numpy as np
import gridfs
import cPickle

class CollData(object):
    '''
    Enables access to all the data within a tfutils collection in form of a
    dictionary. This includes both data stored in the database itself as well
    as in the grid file system. Data is only loaded once and then cached.
    '''
    def __init__(self, host, port, database, collection):
        self.host = host
        self.port = port
        self.database = database
        self.collection = '%s.files' % collection
        self._collection_client = pm.MongoClient(port = self.port, host = self.host)[self.database][self.collection]
        self._exps = {}
        self.keys = self._exps.keys
        
        self._add_exp(self._collection_client.distinct('exp_id'))
        
        print('Collection [\'%s\'][\'%s\'] contains \'exp_id\':' % (self.database, self.collection))
        print(self.keys())
    
    def _add_exp(self, exp_ids):
        if isinstance(exp_ids, str):
            exp_ids = [exp_ids]
        for exp_id in exp_ids:
            if exp_id not in self._exps:
                self._exps[exp_id] = ExpData(exp_id, self._collection_client)

    def __repr__(self):
	return str(self.keys())
               
    def __getitem__(self, exp):
        return self._exps[exp]
        
        
class ExpData(object):
    '''
    Enables access to all the data within a tfutils experiment with experiment
    id exp_id in form of a dictionary. This includes both data stored in the
    database itself as well as in the grid file system. Data is only loaded
    once and then cached.
    '''
    def __init__(self, exp_id, collection_client):
        self._collection_client = collection_client
        self._vals = {}
        self._gfs = {}
        self._fetched_ids = {}
        self.exp_id = exp_id

    def keys(self):
        if not hasattr(self, '_keys'):
            self._keys = self._fetch_keys()
        return self._keys

    def __repr__(self):
	return str(self.keys())

    def __getitem__(self, key): 
        if key not in self._vals:
            self._fetch_data(key)
        return self._vals[key]
    
    def _postproc_results(self, results, ids):
        self._gfs, self._gfs_keys = self._fetch_gfs_data(ids)
        for res, idx in zip(results, ids):
            for r in res:
                if isinstance(res, list):
                    r['gfs'] = GfsDictWrapper(self._gfs[idx], self._gfs_keys, 'train_results')
                elif isinstance(res, dict):
                    res[r]['gfs'] = GfsDictWrapper(self._gfs[idx], self._gfs_keys, 'validation_results', r)
        return results
    
    def _fetch_keys(self):
        keys = []
        for entry in self._collection_client.find({'exp_id': self.exp_id}):
            for key in entry:
                if key not in keys:
                    keys.append(key)
        return keys

    def _fetch_data(self, key):
        if key not in self._vals:
            self._vals[key] = []
            self._fetched_ids[key] = []
        records = [record for record in self._collection_client.find(
                {'exp_id': self.exp_id, key: {'$exists' : True}, 
                 '_id': {'$nin' : self._fetched_ids[key]}}, projection = [key])]
        self._vals[key].extend([record[key] for record in records])
        self._fetched_ids[key].extend([record['_id'] for record in records])
        if key in ['train_results', 'validation_results'] and len(self._vals[key]) > 0:
            self._vals[key] = self._postproc_results(self._vals[key], self._fetched_ids[key])
        return self._vals[key]
 
    def _fetch_gfs_data(self, ids):
        if not hasattr(self, '_gfs_keys'):
            self._gfs_keys = self.__getitem__('params')[0]['save_params']['save_to_gfs']
        for idx in ids:
            if idx not in self._gfs:
                self._gfs[idx] = GfsData(self.exp_id, self._collection_client, idx)
        return self._gfs, self._gfs_keys
    
    def update(self):
        self._keys = self._fetch_keys()
        for key in self._vals:
            self._fetch_data(key)

        
class GfsData(object):
    '''
    Loads data from the grid file system. Data is loaded only once and then
    cached.
    '''
    def __init__(self, exp_id, collection_client, _id):
        self._exp_id = exp_id
        self._collection_client = collection_client
        self._id = _id
    
    def load_from_gfs(self):
        if not hasattr(self, '_vals'):
            records = self._collection_client.find({'item_for' : self._id})
            if records.count() > 0:
                fn = records[0]['filename']
                fs = gridfs.GridFS(self._collection_client.database, 
                        self._collection_client.collection.name[:-17])
                fh = fs.get_last_version(fn)
                self._vals = cPickle.loads(fh.read())
                fh.close()
            else:
                self._vals = {}
        return self._vals
     
        
class GfsDictWrapper(object):
    '''
    Wraps the GfsData class to allow accessing subkeys within the top level
    gridfs keys.
    '''
    def __init__(self, gfs_data, gfs_keys, results_key, subresults_key = None):
        self._gfs_data = gfs_data
        self._gfs_keys = gfs_keys
        self._results_key = results_key
        self._subresults_key = subresults_key

    def __repr__(self):
	strs = []
        data = self._subselect_data()
	for k in data:
	    strs.append(str(data[k]))
	return '[%s]' % ', '.join(strs)

    def __getitem__(self, key):
        data = self._subselect_data()
        return data[key]

    def _subselect_data(self):
        data = self._gfs_data.load_from_gfs()
        if len(data.keys()) > 0:
            data = data[self._results_key]
            if self._subresults_key is not None:
                data = data[self._subresults_key]
                self._gfs_keys = data.keys()
        return data
        
    def keys(self):
        return self._gfs_keys
        
    
if __name__ == '__main__':
    coll = CollData(host='localhost',
                    port=24444,
                    database='future_prediction',
                    collection='particles')
