from pytorch_pfn_extras.training.extensions import snapshot_writers  # NOQA
from pytorch_pfn_extras.training.extensions._snapshot import snapshot  # NOQA
from pytorch_pfn_extras.training.extensions._snapshot import snapshot_object  # NOQA
from pytorch_pfn_extras.training.extensions.best_value import BestValue  # NOQA
from pytorch_pfn_extras.training.extensions.best_value import MaxValue  # NOQA
from pytorch_pfn_extras.training.extensions.best_value import MinValue  # NOQA
from pytorch_pfn_extras.training.extensions.evaluator import Evaluator, DistributedEvaluator, IgniteEvaluator  # NOQA
from pytorch_pfn_extras.training.extensions.fail_on_non_number import FailOnNonNumber  # NOQA
from pytorch_pfn_extras.training.extensions.log_report import LogReport  # NOQA
from pytorch_pfn_extras.training.extensions.lr_scheduler import LRScheduler  # NOQA
from pytorch_pfn_extras.training.extensions.micro_average import MicroAverage  # NOQA
from pytorch_pfn_extras.training.extensions.profile_report import ProfileReport  # NOQA
from pytorch_pfn_extras.training.extensions.parameter_statistics import ParameterStatistics  # NOQA
from pytorch_pfn_extras.training.extensions.plot_report import PlotReport  # NOQA
from pytorch_pfn_extras.training.extensions.profile_report import ProfileReport  # NOQA
from pytorch_pfn_extras.training.extensions.slack import Slack, SlackWebhook  # NOQA
from pytorch_pfn_extras.training.extensions.value_observation import observe_lr  # NOQA
from pytorch_pfn_extras.training.extensions.value_observation import observe_value  # NOQA
from pytorch_pfn_extras.training.extensions.variable_statistics_plot import VariableStatisticsPlot  # NOQA
from pytorch_pfn_extras.training.extensions import util as _util

from pytorch_pfn_extras.training.extensions.print_report import PrintReport as PrintReportCLI  # NOQA
from pytorch_pfn_extras.training.extensions.progress_bar import ProgressBar as ProgressBarCLI  # NOQA

try:
    from pytorch_pfn_extras.training.extensions.print_report_notebook import PrintReportNotebook  # NOQA
    from pytorch_pfn_extras.training.extensions.progress_bar_notebook import ProgressBarNotebook  # NOQA
    _ipython_module_available = True
except ImportError:
    _ipython_module_available = False


if _util._is_notebook():
    PrintReport = PrintReportNotebook
    ProgressBar = ProgressBarNotebook
else:
    PrintReport = PrintReportCLI  # type: ignore[assignment,misc]
    ProgressBar = ProgressBarCLI  # type: ignore[assignment,misc]
