Quick Start
===========

First, pytorch-pfn-extras organizes the training code 
implemented using PyTorch using the Trainer/Evaluator classes.

Next, it provides the following interfaces for the organized PyTorch training code.

1. Addition of extensions for analysis and visualization
2. Runtime changes
3. Addition of custom training steps
4. Custom data handling

Step 1: Use Trainer
-------------------

First, combine the Model and Optimizer you want to learn into the Trainer.

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

Finally, save the trained model obtained through training.

.. literalinclude:: /_example/quick_start_save.py
    :language: python
    :caption: quick_start_save.py

The saved model parameters are stored in a file path including the timing they were saved under ``./result``.

Snapshots are generated using ``state_dict()``. Please refer to the official PyTorch `docs <https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference>`_ for how to load the model.
