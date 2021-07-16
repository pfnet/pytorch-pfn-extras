import tempfile

from pytorch_pfn_extras import logging


def test_file_output():
    try:
        with tempfile.NamedTemporaryFile() as logfile:
            logfile.close()  # this is needed for Windows
            logging._configure_logging(filename=logfile.name, level='DEBUG')
            logger = logging._get_root_logger()
            logger.info('TEST LOG MESSAGE')
            with open(logfile.name) as f:
                assert 'TEST LOG MESSAGE' in f.read()
    finally:
        logging._configure_logging()


def test_get_logger():
    logger = logging.get_logger('app')
    logger.setLevel(logging.DEBUG)
    assert logger.name == 'ppe.app'
    assert logger.level == logging.DEBUG
