import logging
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
