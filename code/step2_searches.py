import datetime
import logging
import time

import pandas as pd

from searches.surprise_search import SurpriseSearch
from settings.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from utils.machine_information import machine_information
from utils.read_from_terminal import read_input_to_searching

logger = logging.getLogger(__name__)


def main():
    """
    Main function to start the Random Search
    """
    experimental_setup = read_input_to_searching()
    # Setup Log configuration
    setup_logging(log_error="error.log", log_info="info.log",
                  save_path=PathDirFile.set_log_search_path(recommender=experimental_setup['recommender'],
                                                            dataset=experimental_setup['dataset']))
    logger.info("$" * 50)
    # Logging machine data
    machine_information()
    logger.info("-" * 50)
    logger.info("SEARCH FOR THE BEST PARAMETER VALUES")
    logger.info(" ".join(['>>', 'Recommender:', experimental_setup['recommender']]))
    logger.info(" ".join(['>>', 'Dataset:', experimental_setup['dataset']]))
    logger.info("$" * 50)
    # Starting the counter
    start_time = time.time()
    logger.info('ooo start at ' + time.strftime('%H:%M:%S'))
    # Executing the Random Search
    search_instance = SurpriseSearch(recommender=experimental_setup['recommender'],
                                     dataset=experimental_setup['dataset'])
    search_instance.fit()
    # Finishing the counter
    finish_time = time.time()
    logger.info('XXX stop at ' + time.strftime('%H:%M:%S'))
    total_time = datetime.timedelta(seconds=finish_time - start_time)
    time_df = pd.DataFrame({"stated_at": [start_time], "finished_at": [finish_time], "total": [total_time]})
    # Saving execution time
    time_df.to_csv(PathDirFile.set_search_time_file(dataset=experimental_setup['dataset'],
                                                    recommender=experimental_setup['recommender']))
    # Finishing the Step
    logger.info(" ".join(['->>', 'Time Execution:', str(total_time)]))
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))


if __name__ == '__main__':
    """
    Start the Random Search
    """
    main()
