import logging
import os
import platform
import socket

from settings.constants import Constants

logger = logging.getLogger(__name__)


def machine_information():
    """

    :return:
    """
    node = '' or platform.node() or socket.gethostname() or os.uname().nodename
    logger.info("> MACHINE INFORMATION")
    logger.info(" ".join(['>>', 'N Cores:', str(Constants.N_CORES)]))
    logger.info(" ".join(['>>', 'Machine RAM:', str(Constants.MEM_RAM)]))
    logger.info(" ".join(['>>', 'Machine Name:', node]))
