import json 
import datetime
import inspect

import numpy as np
from bson.objectid import ObjectId
import git

from tfutils.error import RepoIsDirtyError


def jsonize(x):
    try:
        json.dumps(x)
    except TypeError:
        return SONify(x)
    else:
        return x
        

def git_check_and_info(srcpath):
    repo = git.Repo(srcpath, search_parent_directories=True)
    if repo.is_dirty():
    	raise RepoIsDirtyError('repo at %s is dirty' % repo.git_dir)
    branchname = repo.active_branch.name
    commit = repo.active_branch.commit.hexsha
    origin = repo.remote('origin')
    urls = map(str, list(origin.urls))
    remote_ref = [_r for _r in origin.refs if _r.name == 'origin/' + branchname]
    assert len(remote_ref) > 0, 'Active branch %s not in origin ref' % branchname
    remote_ref = remote_ref[0]
    log = remote_ref.log()
    shas = [_r.oldhexsha for _r in log] + [_r.newhexsha for _r in log]
    assert commit in shas, 'Commit %s not in remote origin log for branch %s' % (commit,
                                                                            branchname)
    info = {'srcpath': srcpath,
    	    'git_dir': repo.git_dir,
    	    'active_branch': branchname,
    	    'commit': commit, 
    	    'remote_urls': urls}
    return info
    	    
    	    
    	    
def make_mongo_safe(_d):
	klist = _d.keys()[:]
	for _k in klist:
		if hasattr(_d[_k], 'keys'):
			print(_k)
			make_mongo_safe(_d[_k])
		if '.' in _k:
			_d[_k.replace('.', '___')] = _d.pop(_k)

def SONify(arg, memo=None):
    if memo is None:
        memo = {}
    if id(arg) in memo:
        rval = memo[id(arg)]
    if isinstance(arg, ObjectId):
        rval = arg
    elif isinstance(arg, datetime.datetime):
        rval = arg
    elif isinstance(arg, np.floating):
        rval = float(arg)
    elif isinstance(arg, np.integer):
        rval = int(arg)
    elif isinstance(arg, (list, tuple)):
        rval = type(arg)([SONify(ai, memo) for ai in arg])
    elif isinstance(arg, collections.OrderedDict):
        rval = collections.OrderedDict([(SONify(k, memo), SONify(v, memo))
            for k, v in arg.items()])
    elif isinstance(arg, dict):
        rval = dict([(SONify(k, memo), SONify(v, memo))
            for k, v in arg.items()])
    elif isinstance(arg, (basestring, float, int, type(None))):
        rval = arg
    elif isinstance(arg, np.ndarray):
        if arg.ndim == 0:
            rval = SONify(arg.sum())
        else:
            rval = map(SONify, arg) # N.B. memo None
    # -- put this after ndarray because ndarray not hashable
    elif arg in (True, False):
        rval = int(arg)
    elif callable(arg):
    	modname = inspect.getmodule(arg).__name__
    	objname = arg.__name__
    	srcpth = inspect.getsourcefile(arg)
    	rval = git_check_and_info(srcpth) 
    	rval.update({'objname': objname,
    			     'modname': modname})   	
    	rval = SONify(rval)
    else:
        raise TypeError('SONify', arg)
    memo[id(rval)] = rval
    return rval
