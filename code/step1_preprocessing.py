import datetime
import logging
import time

import pandas as pd

from datasets.registred_datasets import RegisteredDataset
from settings.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from utils.machine_information import machine_information
from utils.read_from_terminal import read_input_to_preprocessing

logger = logging.getLogger(__name__)


def main():
    """
    Main function to start the preprocessing step
    """
    experimental_setup = read_input_to_preprocessing()
    # Setup Log configuration
    setup_logging(log_error="error.log", log_info="info.log",
                  save_path=PathDirFile.set_log_preprocessing_path(dataset=experimental_setup['dataset']))
    # Logging machine data
    logger.info("$" * 50)
    machine_information()
    logger.info("-" * 50)
    # Logging the experiment setup
    logger.info("> DATASET (PREPROCESSING STEP)")
    logger.info(" ".join(['>>', 'Dataset:', experimental_setup['dataset']]))
    logger.info(" ".join(['>>', 'K Folds:', str(experimental_setup['n_folds'])]))
    logger.info(" ".join(['>>', 'Total of Trials:', str(experimental_setup['n_trials'])]))
    logger.info("$" * 50)
    # Starting the counter
    start_time = time.time()
    logger.info('ooo start at ' + time.strftime('%H:%M:%S'))
    # Executing the pre-processing
    RegisteredDataset.preprocessing(dataset=experimental_setup['dataset'], n_trials=experimental_setup['n_trials'],
                                    n_folds=experimental_setup['n_folds'])
    # Finishing the counter
    finish_time = time.time()
    logger.info('XXX stop at ' + time.strftime('%H:%M:%S'))
    # Getting execution time
    execution_time = datetime.timedelta(seconds=finish_time - start_time)
    time_df = pd.DataFrame({"stated_at": [start_time], "finished_at": [finish_time], "total": [execution_time]})
    # Saving execution time
    time_df.to_csv(PathDirFile.set_preprocessing_time_file(dataset=experimental_setup['dataset']))
    # Finishing the step
    logger.info(" ".join(['->>', 'Time Execution:', str(execution_time)]))
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))


if __name__ == '__main__':
    """
    Start the pre-processing step
    """
    main()
