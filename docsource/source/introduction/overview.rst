Overview
========

The core functionality TFUtils currently provides are for saving and loading results of tensorflow model training and validation runs.  (Currently, the backend database for TFUtils is MongoDB but in the future this may be user-specifiable). In many ways, TFUtils provides similar functionality to part of what is provided by the "tensorboard" utility that is part of the basic tensorflow package.  However, we feel that the design choices in TFUtils may be better suited to using tensorflow in  scientific workflows.

The two basic entry point functions are:

``base.train_from_params``
~~~~~~~~~~~~~~~~~~~~~~~~~~

This function is the interface for performing training.  It takes a set of arguments describing how to construct a tensorflow model, a loss function, and learning rate, and an optimizer, as well as arguments describing hot to load a training dataset.  It also takes arguments for describing where (that is, which mongodb instance) to save results to.   The interface will then create the model, run the training,a nd store results intermittently.   The interface also allows for restarting training from an existing point.  The nosetest in tfutils/tests/test.py:test_training illustrates how the train function works.

``base.test_from_params``
~~~~~~~~~~~~~~~~~~~~~~~~~~

This function is the interface for performing testing validation and feature extraction with a pre-trained model.  It takes a set of arguments describing how to load model, together with a set of specifications describing what type of validation to run -- this can either be a computing test performance on a new dataset, or extracting features on an input-by-input basis. The function then loads the model, performs the validation, and saves the results to the database.   The `nosetest tfutils/tests/test.py:test_validation` illustrates how to use the test function to obtain performance of pre-trained model on a new dataset.   The `nosetest tfutils/tests/test.py:test_feature_extraction` illustrates how to use the test function to extract features using a pre-trained model.

The docstrings of these functions, as well as the test code test, provide much more detailed information on how to use the libraries.
