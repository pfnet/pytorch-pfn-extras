import logging
import os

from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # NOQA
from typing import Optional

_logger_name = 'ppe'
_logger_format = '[%(name)s] %(asctime)s: (%(levelname)s) %(message)s'
_logger = None


def _configure_logging(
        *,
        filename: Optional[str] = None,
        level: str = 'ERROR',
        format: str = _logger_format
) -> None:
    global _logger
    filename = os.environ.get('PPE_LOG_FILENAME', filename)
    if filename is None:
        handler: logging.Handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter(format))
    # To dynamically change the level if needed
    # basicConfig does not allow to change the level right after
    _logger = logging.getLogger(_logger_name)
    level = os.environ.get('PPE_LOG_LEVEL', level)
    for lvl in (logging.DEBUG, logging.INFO,
                logging.WARNING, logging.ERROR, logging.CRITICAL):
        if logging.getLevelName(lvl) == level:
            _logger.setLevel(lvl)
            break
    else:
        _logger.setLevel(logging.INFO)
        _logger.warning('invalid PPE_LOG_LEVEL (%s); using INFO', level)
    _logger.addHandler(handler)


def _get_root_logger() -> logging.Logger:
    """Returns a logger to be used by pytorch-pfn-extras."""
    assert _logger is not None
    return _logger


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger to be used by applications.

    Args:
        name (str): Name used to register and retrieve the logger object.

    Returns:
        A logging.Logger object used to log in the application code.
    """
    return _get_root_logger().getChild(name)
