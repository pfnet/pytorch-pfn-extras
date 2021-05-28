import logging
import os
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
    filename = os.environ.get('PPE_LOG_FILENAME', filename)
    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    # To dynamically change the level if needed
    # basicConfig does not allow to change the level right after
    logger = logging.getLogger('ppe')
    level = os.environ.get('PPE_LOG_LEVEL', level)
    for lvl in (logging.DEBUG, logging.INFO,
                logging.WARNING, logging.ERROR, logging.CRITICAL):
        if logging.getLevelName(lvl) == level:
            handler.setLevel(lvl)
            break
    else:
        logger.warning('invalid PPE_LOG_LEVEL (%s); continue with INFO', level)
        handler.setLevel(logging.INFO)
    logger.addHandler(handler)


configureLogging()
