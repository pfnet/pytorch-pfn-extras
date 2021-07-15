import logging
import os

_logger_name = 'ppe'
_logger = None


def _configure_logging():
    global _logger
    filename = os.environ.get('PPE_LOG_FILENAME', None)
    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(filename)
    # To dynamically change the level if needed
    # basicConfig does not allow to change the level right after
    _logger = logging.getLogger(_logger_name)
    level = os.environ.get('PPE_LOG_LEVEL', 'ERROR')
    for lvl in (logging.DEBUG, logging.INFO,
                logging.WARNING, logging.ERROR, logging.CRITICAL):
        if logging.getLevelName(lvl) == level:
            _logger.setLevel(lvl)
            break
    else:
        _logger.setLevel(logging.INFO)
        _logger.warning('invalid PPE_LOG_LEVEL (%s); using INFO', level)
    _logger.addHandler(handler)


def get_logger(name=None):
    if name is None:
        return _logger
    else:
        return _logger.getChild(name)
