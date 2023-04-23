import logging

from joblib import Parallel, delayed

from processing.surprise_recommender_algorithms import SurpriseRecommenderAlgorithm
from settings.constants import Constants
from utils.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep3(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step3()
        print(self.experimental_settings)

    @staticmethod
    def set_the_logfile_by_instance(dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_processing_path(
                algorithm=algorithm,
                trial=trial,
                fold=fold,
                dataset=dataset
            )
        )

    def print_basic_info_by_instance(self, dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[PROCESSING STEP] - RECOMMENDER ALGORITHM")
        logger.info(" ".join(['>>', 'Dataset:', dataset]))
        logger.info(" ".join(['>>', 'Trial:', str(trial)]))
        logger.info(" ".join(['>>', 'Fold:', str(fold)]))
        logger.info(" ".join(['>>', 'Algorithm:', algorithm]))
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """
        self.recommender_parallelization()

    def recommender_parallelization(self):
        """
        Main method to start the processing step in parallel.
        """
        # Starting the recommender algorithm
        Parallel(n_jobs=Constants.N_CORES)(delayed(self.starting_recommender)(
            recommender=self.experimental_settings['recommender'],
            dataset=self.experimental_settings['dataset'],
            trial=trial, fold=fold
        ) for fold in self.experimental_settings['fold'] for trial in self.experimental_settings['trial'])
        # Finishing the Step
        logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))

    def starting_recommender(self, dataset: str, recommender: str, trial: int, fold: int):
        """
        Function to starting the recommender algorithm.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        """
        self.set_the_logfile_by_instance(
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        )
        self.print_basic_info_by_instance(
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        )

        # Starting the counter
        self.start_count()

        # Executing the processing step
        recommender_algorithm = SurpriseRecommenderAlgorithm(
            dataset_name=dataset, trial=trial, fold=fold, recommender_name=recommender
        )
        recommender_algorithm.run()

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_processing_time(
            data=self.clock_data(),
            dataset=dataset, trial=trial, fold=fold, algorithm=recommender
        )
        # Finishing the step
        logger.info(" ".join(['->>', 'Time Execution:', str(self.get_total_time())]))


if __name__ == '__main__':
    """
    It starts the processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep3()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
