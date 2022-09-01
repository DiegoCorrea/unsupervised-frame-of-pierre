import datetime
import logging
import time

import pandas as pd
from joblib import Parallel, delayed

from processing.surprise_recommender_algorithms import SurpriseRecommenderAlgorithm
from settings.constants import Constants
from settings.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from utils.machine_information import machine_information
from utils.read_from_terminal import read_input_to_processing

logger = logging.getLogger(__name__)


def starting_processing(dataset: str, recommender: str, trial: int, fold: int):
    """
    Function to starting the processing and run the recommender algorithm.

    :param dataset: A string that's representing the dataset name.
    :param recommender: A string that's representing the recommender algorithm name.
    :param trial: The trial number.
    :param fold: The fold number.
    """
    # Setup Log configuration
    setup_logging(log_error="error.log", log_info="info.log",
                  save_path=PathDirFile.set_log_processing_path(dataset=dataset, recommender=recommender,
                                                           trial=trial, fold=fold))
    logger.info("$" * 50)
    # Logging machine data
    machine_information()
    logger.info("-" * 50)
    # Logging the experiment setup
    logger.info("RECOMMENDER ALGORITHM (PROCESSING STEP)")
    logger.info(" ".join(['>>', 'Recommender:', recommender]))
    logger.info(" ".join(['>>', 'Dataset:', dataset]))
    logger.info(" ".join(['>>', 'Trial:', str(trial)]))
    logger.info(" ".join(['>>', 'Fold:', str(fold)]))
    logger.info("$" * 50)
    # Starting the counter
    start_time = time.time()
    logger.info('ooo start at ' + time.strftime('%H:%M:%S'))
    # Executing the processing step
    recommender_algorithm = SurpriseRecommenderAlgorithm(dataset_name=dataset, trial=trial,
                                                         recommender_name=recommender, fold=fold)
    recommender_algorithm.run()
    # Finishing the counter
    finish_time = time.time()
    logger.info('XXX stop at ' + time.strftime('%H:%M:%S'))
    total_time = datetime.timedelta(seconds=finish_time - start_time)
    time_df = pd.DataFrame({"stated_at": [start_time], "finished_at": [finish_time], "total": [total_time]})
    # Saving execution time
    time_df.to_csv(
        PathDirFile.set_processing_time_file(dataset=dataset, trial=trial, recommender=recommender, fold=fold))


def main():
    """
    Main function to start the processing step
    """
    experimental_setup = read_input_to_processing()
    # Starting the recommender algorithm
    Parallel(n_jobs=Constants.N_CORES)(
        delayed(starting_processing)(recommender=experimental_setup['recommender'],
                                     dataset=experimental_setup['dataset'],
                                     trial=trial + 1, fold=fold + 1) for fold in
        range(Constants.K_FOLDS_VALUE) for trial in range(Constants.N_TRIAL_VALUE))
    # Finishing the Step
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))


if __name__ == '__main__':
    """
    Start the processing step
    """
    main()
