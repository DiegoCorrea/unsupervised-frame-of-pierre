import datetime
import logging
import time
import os
import platform
import socket
from pandas import DataFrame

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
        self.time_data_df = None

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

    def clock_data(self) -> DataFrame:
        total_time = datetime.timedelta(seconds=self.get_total_time())
        self.time_data_df = DataFrame({
            "stated_at": [self.get_start_time()],
            "finished_at": [self.get_finish_time()],
            "total": [total_time]
        })
        return self.time_data_df

    def get_finish_time(self):
        return self.finish_time

    def get_total_time(self):
        return self.finish_time - self.start_time

