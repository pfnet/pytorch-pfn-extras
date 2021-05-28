import logging
import os
import sys

from pytorch_pfn_extras import config  # NOQA
from pytorch_pfn_extras import cuda  # NOQA
from pytorch_pfn_extras import dataset  # NOQA
from pytorch_pfn_extras import dataloaders  # NOQA
from pytorch_pfn_extras import nn  # NOQA
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


def configureLogging(filename=None, level=logging.ERROR):

    filename = os.environ.get('PPE_LOGGING_FILENAME', filename)
    if filename is None:
        logging.basicConfig(stream=sys.stdout)
    else:
        logging.basicConfig(filename=filename)
    # To dynamically change the level if needed
    # basicConfig does not allow to change the level right after
    level = os.environ.get('PPE_LOGGING_LEVEL', level)
    logging.getLogger().setLevel(level)


configureLogging()
