"""
Entrance of tfutils

Check `tfutils.train` for function `train_from_params`.

Check `tfutils.test` for function `test_from_params`.
"""

# Main function for training
from tfutils.train import train_from_params

# Main function for testing
from tfutils.test import test_from_params

# One helper function for previous tfutils users
from tfutils.helper import get_params
