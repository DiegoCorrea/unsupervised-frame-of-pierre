import logging
import time
import os
import platform
import socket

from settings.constants import Constants

logger = logging.getLogger(__name__)


class Step:
    """
    TODO: Docstring
    """

    def __init__(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = None
        self.start_time = None
        self.finish_time = None

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        pass

    def set_the_logfile(self):
        """
        TODO: Docstring
        """
        pass

    def print_basic_info(self):
        """
        TODO: Docstring
        """
        pass

    @staticmethod
    def machine_information():
        """
        TODO: Docstring
        """
        node = '' or platform.node() or socket.gethostname() or os.uname().nodename
        logger.info("> MACHINE INFORMATION")
        logger.info(" ".join(['>>', 'N Cores:', str(Constants.N_CORES)]))
        logger.info(" ".join(['>>', 'Machine RAM:', str(Constants.MEM_RAM)]))
        logger.info(" ".join(['>>', 'Machine Name:', node]))

    def start_count(self):
        self.start_time = time.time()
        logger.info('ooo start at ' + time.strftime('%H:%M:%S'))

    def get_start_time(self):
        return self.start_time

    def finish_count(self):
        self.finish_time = time.time()
        logger.info('XXX stop at ' + time.strftime('%H:%M:%S'))

    def get_finish_time(self):
        return self.finish_time

    def get_total_time(self):
        return self.finish_time - self.start_time

