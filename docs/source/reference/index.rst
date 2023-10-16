
API Reference
=============

* :ref:`genindex`

Package
-------
.. autosummary::
   :toctree: generated/
   :recursive:
   
   pytorch_pfn_extras

.. currentmodule:: pytorch_pfn_extras

Training Loop
------------------

Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   engine.create_trainer
   engine.create_evaluator
   handler.BaseLogic
   handler.Logic
   handler.BaseHandler
   handler.Handler
   runtime.BaseRuntime
   runtime.PyTorchRuntime


Extensions Manager
~~~~~~~~~~~~~~~~~~

.. autosummary::

   training.ExtensionsManager
   training.IgniteExtensionsManager

Extensions
~~~~~~~~~~

.. autosummary::

   training.extension.make_extension
   training.extension.Extension
   training.extension.ExtensionEntry


.. autosummary::

   training.extensions.BestValue
   training.extensions.Evaluator
   training.extensions.LogReport
   training.extensions.MaxValue
   training.extensions.MicroAverage
   training.extensions.MinValue
   training.extensions.observe_lr
   training.extensions.observe_value
   training.extensions.ParameterStatistics
   training.extensions.PlotReport
   training.extensions.PrintReport
   training.extensions.ProgressBar
   training.extensions.ProfileReport
   training.extensions.snapshot
   training.extensions.Slack
   training.extensions.SlackWebhook
   training.extensions.VariableStatisticsPlot

Triggers
~~~~~~~~

.. autosummary::

   training.triggers.EarlyStoppingTrigger
   training.triggers.IntervalTrigger
   training.triggers.ManualScheduleTrigger
   training.triggers.BestValueTrigger
   training.triggers.MaxValueTrigger
   training.triggers.MinValueTrigger
   training.triggers.OnceTrigger
   training.triggers.TimeTrigger


Reporting
~~~~~~~~~

.. autosummary::

   reporting.Reporter
   reporting.report
   reporting.report_scope


Logging
~~~~~~~

.. autosummary::

   logging.get_logger

Profiler
~~~~~~~~

.. autosummary::

   profiler.TimeSummary.report

Distributed Training
---------------------

.. autosummary::

   nn.parallel.DistributedDataParallel
   distributed.initialize_ompi_environment


Check Pointing
---------------------

.. autosummary::

   utils.checkpoint


Lazy Modules
------------------

.. autosummary::

   nn.Ensure
   nn.ensure
   nn.LazyLinear
   nn.LazyConv1d
   nn.LazyConv2d
   nn.LazyConv3d
   nn.LazyBatchNorm1d
   nn.LazyBatchNorm2d
   nn.LazyBatchNorm3d


ONNX
------------------

Export
~~~~~~~

.. autosummary::

   onnx.export
   onnx.export_testcase


Annotation
~~~~~~~~~~~

.. autosummary::

   onnx.annotate
   onnx.apply_annotation
   onnx.scoped_anchor
   onnx.export
   onnx.export_testcase


Datasets
------------------------

.. autosummary::

   dataset.SharedDataset
   dataset.TabularDataset
   dataset.ItemNotFoundException


Config
------------------------

.. autosummary::

   config.Config

.. autosummary::

   config_types.optuna_types
   config_types.load_path_with_optuna_types


NumPy/CuPy Compatibility
------------------------

.. autosummary::

   from_ndarray
   as_ndarray
   get_xp
   as_numpy_dtype
   from_numpy_dtype

.. autosummary::

   cuda.stream
   cuda.use_torch_mempool_in_cupy
   cuda.use_default_mempool_in_cupy
