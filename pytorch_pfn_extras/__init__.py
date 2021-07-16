# Configure the logging before instantiating anything else
from pytorch_pfn_extras import logging  # NOQA
logging._configure_logging()

from pytorch_pfn_extras import config  # NOQA
from pytorch_pfn_extras import cuda  # NOQA
from pytorch_pfn_extras import dataset  # NOQA
from pytorch_pfn_extras import dataloaders  # NOQA
from pytorch_pfn_extras import nn  # NOQA
from pytorch_pfn_extras import profiler  # NOQA
from pytorch_pfn_extras import reporting  # NOQA
from pytorch_pfn_extras import training  # NOQA
from pytorch_pfn_extras import utils  # NOQA
from pytorch_pfn_extras import writing  # NOQA

from pytorch_pfn_extras._tensor import from_ndarray  # NOQA
from pytorch_pfn_extras._tensor import as_ndarray  # NOQA
from pytorch_pfn_extras._tensor import get_xp  # NOQA
from pytorch_pfn_extras._tensor import as_numpy_dtype  # NOQA
from pytorch_pfn_extras._tensor import from_numpy_dtype  # NOQA

from pytorch_pfn_extras._version import __version__  # NOQA
