Quick Start
===========

First, pytorch-pfn-extras organizes the training code 
implemented using PyTorch using the Trainer/Evaluator classes.

Next, it provides the following interfaces for training PyTorch models.

1. Addition of extensions for analysis and visualization
2. Runtime changes
3. Addition of custom training steps
4. Custom data handling

Step 1: Use Trainer
-------------------

First, pass to the Trainer the Model and Optimizer you want to train.

.. literalinclude:: /_example/quick_start_trainer.py
    :language: python
    :caption: quick_start_trainer.py


Step 2: Get Log
---------------

Next, collect the logs of the training progress.

.. literalinclude:: /_example/quick_start_log.py
    :language: python
    :caption: quick_start_log.py

The logs of the collected learning progress are output to ``./result/log``.

Step 3: Display of progress
---------------------------

Make it possible to check the progress of the learning.

.. literalinclude:: /_example/quick_start_progress.py
    :language: python
    :caption: quick_start_progress.py

Step 4: Save Model
------------------

Finally, save the trained model.

.. literalinclude:: /_example/quick_start_save.py
    :language: python
    :caption: quick_start_save.py

The model parameters are stored with a file name that includes the time they were saved under ``./result``.

Snapshots are generated using ``state_dict()``. Please refer to the official PyTorch `docs <https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference>`_ for how to load the model.
