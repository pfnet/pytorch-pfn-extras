from pytorch_pfn_extras.training.extensions import snapshot_writers  # NOQA
from pytorch_pfn_extras.training.extensions._snapshot import (  # NOQA
    snapshot,
    snapshot_object,
)
from pytorch_pfn_extras.training.extensions.evaluator import (  # NOQA
    Evaluator,
    IgniteEvaluator,
)
from pytorch_pfn_extras.training.extensions.fail_on_non_number import (  # NOQA
    FailOnNonNumber,
)
from pytorch_pfn_extras.training.extensions.log_report import LogReport  # NOQA
from pytorch_pfn_extras.training.extensions.lr_scheduler import (  # NOQA
    LRScheduler,
)
from pytorch_pfn_extras.training.extensions.micro_average import (  # NOQA
    MicroAverage,
)
from pytorch_pfn_extras.training.extensions.parameter_statistics import (  # NOQA
    ParameterStatistics,
)
from pytorch_pfn_extras.training.extensions.plot_report import (  # NOQA
    PlotReport,
)
from pytorch_pfn_extras.training.extensions.print_report import (  # NOQA
    PrintReport,
)
from pytorch_pfn_extras.training.extensions.progress_bar import (  # NOQA
    ProgressBar,
)
from pytorch_pfn_extras.training.extensions.value_observation import (  # NOQA
    observe_lr,
    observe_value,
)
from pytorch_pfn_extras.training.extensions.variable_statistics_plot import (  # NOQA
    VariableStatisticsPlot,
)

try:
    from pytorch_pfn_extras.training.extensions.print_report_notebook import (  # NOQA
        PrintReportNotebook,
    )
    from pytorch_pfn_extras.training.extensions.progress_bar_notebook import (  # NOQA
        ProgressBarNotebook,
    )

    _ipython_module_available = True
except ImportError:
    _ipython_module_available = False
