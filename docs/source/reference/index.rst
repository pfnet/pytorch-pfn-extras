API Reference
=============

* :ref:`genindex`

Package
-------
.. autosummary::
   :toctree: generated/
   :recursive:
   
   pytorch_pfn_extras


Training Loop
------------------

Trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.engine.create_trainer
   pytorch_pfn_extras.engine.create_evaluator
   pytorch_pfn_extras.handler.BaseLogic
   pytorch_pfn_extras.handler.Logic
   pytorch_pfn_extras.handler.BaseHandler
   pytorch_pfn_extras.handler.Handler
   pytorch_pfn_extras.runtime.BaseRuntime
   pytorch_pfn_extras.runtime.PyTorchRuntime


Extensions Manager
~~~~~~~~~~~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.training.ExtensionsManager
   pytorch_pfn_extras.training.IgniteExtensionsManager

Extensions
~~~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.training.extension.make_extension
   pytorch_pfn_extras.training.extension.Extension
   pytorch_pfn_extras.training.extension.ExtensionEntry


.. autosummary::

   pytorch_pfn_extras.training.extensions.BestValue
   pytorch_pfn_extras.training.extensions.Evaluator
   pytorch_pfn_extras.training.extensions.LogReport
   pytorch_pfn_extras.training.extensions.MaxValue
   pytorch_pfn_extras.training.extensions.MicroAverage
   pytorch_pfn_extras.training.extensions.MinValue
   pytorch_pfn_extras.training.extensions.observe_lr
   pytorch_pfn_extras.training.extensions.observe_value
   pytorch_pfn_extras.training.extensions.ParameterStatistics
   pytorch_pfn_extras.training.extensions.PlotReport
   pytorch_pfn_extras.training.extensions.PrintReport
   pytorch_pfn_extras.training.extensions.ProgressBar
   pytorch_pfn_extras.training.extensions.ProfileReport
   pytorch_pfn_extras.training.extensions.snapshot
   pytorch_pfn_extras.training.extensions.Slack
   pytorch_pfn_extras.training.extensions.SlackWebhook
   pytorch_pfn_extras.training.extensions.VariableStatisticsPlot

Triggers
~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.training.triggers.EarlyStoppingTrigger
   pytorch_pfn_extras.training.triggers.IntervalTrigger
   pytorch_pfn_extras.training.triggers.ManualScheduleTrigger
   pytorch_pfn_extras.training.triggers.BestValueTrigger
   pytorch_pfn_extras.training.triggers.MaxValueTrigger
   pytorch_pfn_extras.training.triggers.MinValueTrigger
   pytorch_pfn_extras.training.triggers.OnceTrigger
   pytorch_pfn_extras.training.triggers.TimeTrigger


Reporting
~~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.reporting.Reporter
   pytorch_pfn_extras.reporting.report
   pytorch_pfn_extras.reporting.report_scope


Logging
~~~~~~~

.. autosummary::

   pytorch_pfn_extras.logging.get_logger

Profiler
~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.profiler.TimeSummary.report

Distributed Training
---------------------

.. autosummary::

   pytorch_pfn_extras.nn.parallel.DistributedDataParallel
   pytorch_pfn_extras.distributed.initialize_ompi_environment


Check Pointing
---------------------

.. autosummary::

   pytorch_pfn_extras.utils.checkpoint


Lazy Modules
------------------

.. autosummary::

   pytorch_pfn_extras.nn.Ensure
   pytorch_pfn_extras.nn.ensure
   pytorch_pfn_extras.nn.LazyLinear
   pytorch_pfn_extras.nn.LazyConv1d
   pytorch_pfn_extras.nn.LazyConv2d
   pytorch_pfn_extras.nn.LazyConv3d
   pytorch_pfn_extras.nn.LazyBatchNorm1d
   pytorch_pfn_extras.nn.LazyBatchNorm2d
   pytorch_pfn_extras.nn.LazyBatchNorm3d


ONNX
------------------

Export
~~~~~~~

.. autosummary::

   pytorch_pfn_extras.onnx.export
   pytorch_pfn_extras.onnx.export_testcase


Annotation
~~~~~~~~~~~

.. autosummary::

   pytorch_pfn_extras.onnx.annotate
   pytorch_pfn_extras.onnx.apply_annotation
   pytorch_pfn_extras.onnx.scoped_anchor
   pytorch_pfn_extras.onnx.export
   pytorch_pfn_extras.onnx.export_testcase


Datasets
------------------------

.. autosummary::

   pytorch_pfn_extras.dataset.SharedDataset
   pytorch_pfn_extras.dataset.TabularDataset
   pytorch_pfn_extras.dataset.ItemNotFoundException


Config
------------------------

.. autosummary::

   pytorch_pfn_extras.config.Config

.. autosummary::

   pytorch_pfn_extras.config_types.optuna_types
   pytorch_pfn_extras.config_types.load_path_with_optuna_types


NumPy/CuPy Compatibility
------------------------

.. autosummary::

   pytorch_pfn_extras.from_ndarray
   pytorch_pfn_extras.as_ndarray
   pytorch_pfn_extras.get_xp
   pytorch_pfn_extras.as_numpy_dtype
   pytorch_pfn_extras.from_numpy_dtype

.. autosummary::

   pytorch_pfn_extras.cuda.stream
   pytorch_pfn_extras.cuda.use_torch_mempool_in_cupy
   pytorch_pfn_extras.cuda.use_default_mempool_in_cupy
