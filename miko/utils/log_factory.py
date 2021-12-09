import os
import logging


class LogFactory(object):
    """
    For logging
    """
    def __init__(self, logger=None, log_dir=""):
        self.log_path = os.getcwd() if log_dir == "" else log_dir
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)
        self.log_name = os.path.join(self.log_path, "miko.log")

        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(self.log_name, delay=True)
        file_handler.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_log(self):
        return self.logger


logger = LogFactory(__name__).get_log()
