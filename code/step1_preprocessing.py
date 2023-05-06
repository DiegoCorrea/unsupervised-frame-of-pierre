import datetime
import itertools
import logging
import time

import pandas as pd
from joblib import Parallel, delayed

from datasets.registred_datasets import RegisteredDataset
from graphics.charts import DatasetChart
from scikit_pierre.classes.genre import genre_probability_approach
from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad
from utils.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from utils.input import Input
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep1(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step1()

    def set_the_logfile(self):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_preprocessing_path(dataset=self.experimental_settings['dataset'])
        )

    def print_basic_info(self):
        """
        TODO: Docstring
        """
        # Logging machine data
        logger.info("$" * 50)
        self.machine_information()
        logger.info("-" * 50)
        # Logging the experiment setup
        logger.info(f"> DATASET (PREPROCESSING STEP) - {self.experimental_settings['opt']}")
        logger.info(" ".join(['>>', 'Dataset:', self.experimental_settings['dataset']]))
        logger.info(" ".join(['>>', 'Number of Folds:', str(self.experimental_settings['n_folds'])]))
        logger.info(" ".join(['>>', 'Number of Trials:', str(self.experimental_settings['n_trials'])]))
        logger.info("$" * 50)

    def create_folds(self):
        """
        TODO: Docstring
        """
        # Starting the counter
        self.start_count()
        # Executing the pre-processing
        RegisteredDataset.preprocessing(
            dataset=self.experimental_settings['dataset'],
            n_trials=self.experimental_settings['n_trials'],
            n_folds=self.experimental_settings['n_folds']
        )

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_preprocessing_time(
            data=self.clock_data(),
            dataset=self.experimental_settings['dataset']
        )

        # Finishing the step
        logger.info(" ".join(['->>', 'Time Execution:', str(self.get_total_time())]))

    def create_charts(self):
        """
        TODO: Docstring
        """
        for dataset_name in RegisteredDataset.DATASET_LIST:
            print(dataset_name)
            dt_chat = DatasetChart(dataset_name)
            dt_chat.item_long_tail()
            dt_chat.genres()

    def create_analyzes(self):
        """
        TODO: Docstring
        """
        dataset_instance = RegisteredDataset.load_dataset(
            dataset=self.experimental_settings['dataset']
        )
        dataset_instance.raw_data_basic_info()
        dataset_instance.clean_data_basic_info()

    def create_distribution(self):
        combination = [
            [self.experimental_settings['dataset']], self.experimental_settings['trial'],
            self.experimental_settings['fold'], [self.experimental_settings['distribution']]
        ]

        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(self.compute_distribution)(
                dataset=dataset, trial=trial, fold=fold, distribution=distribution
            ) for dataset, trial, fold, distribution
            in list(itertools.product(*combination)))

    @staticmethod
    def compute_distribution(dataset, trial, fold, distribution):
        """
        TODO: Docstring
        """
        dataset_instance = RegisteredDataset.load_dataset(dataset)
        dist_func = distributions_funcs_pandas(distribution=distribution)
        items_classes_set = genre_probability_approach(item_set=dataset_instance.get_items())
        users_preference_set = dataset_instance.get_train_transactions(
            trial=trial, fold=fold
        )

        grouped_users_preference_set = users_preference_set.groupby(by=[Label.USER_ID])
        users_pref_dist_df = pd.concat([
            dist_func(
                user_pref_set=user_pref_set,
                item_classes_set=items_classes_set
            ) for user_id, user_pref_set in grouped_users_preference_set
        ])
        SaveAndLoad.save_user_preference_distribution(
            data=users_pref_dist_df, dataset=dataset, fold=fold, trial=trial, distribution=distribution
        )

        logger.info(" ... ".join([
            '->> ', 'Compute Distribution Finished to: ', dataset, distribution, str(trial), str(fold)
        ]))

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.DATASET_CHART:
            self.create_charts()
        elif self.experimental_settings['opt'] == Label.DATASET_ANALYZE:
            self.create_analyzes()
        elif self.experimental_settings['opt'] == Label.DATASET_DISTRIBUTION:
            self.create_distribution()
        else:
            self.create_folds()


if __name__ == '__main__':
    """
    It starts the pre-processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep1()
    step.read_the_entries()
    step.set_the_logfile()
    step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System Shutdown', '+' * 10]))
