.. module:: pytorch_pfn_extras

API Reference
=============

* :ref:`genindex`

Training Loop
------------------

Trainer (techincal preview)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   training.ExtensionsManager
   training.IgniteExtensionsManager

Extensions
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   training.extension.make_extension
   training.extension.Extension
   training.extension.ExtensionEntry


.. autosummary::
   :toctree: generated/

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
   training.extensions.VariableStatisticsPlot

Triggers
~~~~~~~~

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   reporting.Reporter
   reporting.report
   reporting.report_scope


Logging
~~~~~~~

.. autosummary::
   :toctree: generated/

   logging.get_logger

Profiler
~~~~~~~~

.. autosummary::
   :toctree: generated/

   profiler.TimeSummary.report

Distributed Training
---------------------

.. autosummary::
   :toctree: generated/

   nn.parallel.DistributedDataParallel
   distributed.initialize_ompi_environment


Check Pointing
---------------------

.. autosummary::
   :toctree: generated/

   utils.checkpoint


Lazy Modules
------------------

.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   onnx.export
   onnx.export_testcase


Annotation
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   onnx.annotate
   onnx.apply_annotation
   onnx.scoped_anchor
   onnx.export
   onnx.export_testcase


Datasets
------------------------

.. autosummary::
   :toctree: generated/

   dataset.SharedDataset
   dataset.TabularDataset
   dataset.ItemNotFoundException


Config
------------------------

.. autosummary::
   :toctree: generated/

   config.Config

.. autosummary::
   :toctree: generated/

   config_types.optuna_types
   config_types.load_path_with_optuna_types


NumPy/CuPy Compatibility
------------------------

.. autosummary::
   :toctree: generated/

   from_ndarray
   as_ndarray
   get_xp
   as_numpy_dtype
   from_numpy_dtype

.. autosummary::
   :toctree: generated/

   cuda.stream
   cuda.use_torch_mempool_in_cupy
   cuda.use_default_mempool_in_cupy
