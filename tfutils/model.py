"""
Entrance of model building tools.
IMPORTANT thing to know: 
    Tfutils does not require the usage of these tools at all!
    We put these tools here just to be used in:
        tutorials, function tests, and for users who previously used these tools
"""

# Model building tool used in tutorials and function tests
from model_tool import ConvNet as ConvNet_new
from model_tool import mnist_tfutils as mnist_tfutils_new
from model_tool import alexnet_tfutils as alexnet_tfutils_new

# Model building tool used for previous users
# This tool is too complicated to understand and no longer recommended at all!
from model_tool_old import ConvNet, mnist_tfutils, alexnet_tfutils
