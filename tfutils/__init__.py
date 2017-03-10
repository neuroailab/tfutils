import os
from tfutils import data
from tfutils import base
from tfutils import error
from tfutils import model
from tfutils import optimizer
from tfutils import utils

__all__ = [base, data, error, model, optimizer, utils]

# put __version__ in the namespace
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'version.py')).read())

