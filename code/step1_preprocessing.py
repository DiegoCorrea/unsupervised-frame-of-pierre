import datetime
import logging
import time

import pandas as pd

from datasets.registred_datasets import RegisteredDataset
from settings.labels import Label
from settings.logging_settings import setup_logging
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
        start_time = time.time()
        logger.info('ooo start at ' + time.strftime('%H:%M:%S'))
        # Executing the pre-processing
        RegisteredDataset.preprocessing(
            dataset=self.experimental_settings['dataset'],
            n_trials=self.experimental_settings['n_trials'],
            n_folds=self.experimental_settings['n_folds']
        )
        # Finishing the counter
        finish_time = time.time()
        logger.info('XXX stop at ' + time.strftime('%H:%M:%S'))
        # Getting execution time
        execution_time = datetime.timedelta(seconds=finish_time - start_time)
        time_df = pd.DataFrame({"stated_at": [start_time], "finished_at": [finish_time], "total": [execution_time]})
        # Saving execution time
        PathDirFile.save_split_time_file(
            data_df=time_df, dataset=self.experimental_settings['dataset']
        )
        # Finishing the step
        logger.info(" ".join(['->>', 'Time Execution:', str(execution_time)]))

    def create_charts(self):
        """
        TODO: Docstring
        """
        pass

    def create_analyzes(self):
        """
        TODO: Docstring
        """
        pass

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.DATASET_CHART:
            self.create_charts()
        elif self.experimental_settings['opt'] == Label.DATASET_ANALYZE:
            self.create_analyzes()
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
