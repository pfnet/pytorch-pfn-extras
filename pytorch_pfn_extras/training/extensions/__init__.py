from pytorch_pfn_extras.training.extensions import snapshot_writers  # NOQA
from pytorch_pfn_extras.training.extensions._snapshot import snapshot  # NOQA
from pytorch_pfn_extras.training.extensions._snapshot import snapshot_object  # NOQA
from pytorch_pfn_extras.training.extensions.evaluator import Evaluator, IgniteEvaluator  # NOQA
from pytorch_pfn_extras.training.extensions.fail_on_non_number import FailOnNonNumber  # NOQA
from pytorch_pfn_extras.training.extensions.log_report import LogReport  # NOQA
from pytorch_pfn_extras.training.extensions.lr_scheduler import LRScheduler  # NOQA
from pytorch_pfn_extras.training.extensions.micro_average import MicroAverage  # NOQA
from pytorch_pfn_extras.training.extensions.print_report import PrintReport  # NOQA
from pytorch_pfn_extras.training.extensions.profile_report import ProfileReport  # NOQA
from pytorch_pfn_extras.training.extensions.progress_bar import ProgressBar  # NOQA
from pytorch_pfn_extras.training.extensions.parameter_statistics import ParameterStatistics  # NOQA
from pytorch_pfn_extras.training.extensions.plot_report import PlotReport  # NOQA
from pytorch_pfn_extras.training.extensions.profile_report import ProfileReport  # NOQA
from pytorch_pfn_extras.training.extensions.value_observation import observe_lr  # NOQA
from pytorch_pfn_extras.training.extensions.value_observation import observe_value  # NOQA
from pytorch_pfn_extras.training.extensions.variable_statistics_plot import VariableStatisticsPlot  # NOQA

try:
    from pytorch_pfn_extras.training.extensions.print_report_notebook import PrintReportNotebook  # NOQA
    from pytorch_pfn_extras.training.extensions.progress_bar_notebook import ProgressBarNotebook  # NOQA

    _ipython_module_available = True
except ImportError:
    _ipython_module_available = False
