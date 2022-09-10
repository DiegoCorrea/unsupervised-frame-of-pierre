import logging

from searches.cluster_search import UnsupervisedLearning
from searches.recommender_search import SurpriseSearch
from settings.labels import Label
from settings.logging_settings import setup_logging
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep2(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step2()

    def set_the_logfile(self):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_search_path(
                algorithm=self.experimental_settings['recommender'],
                dataset=self.experimental_settings['dataset']
            )
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
        logger.info("-" * 50)
        logger.info("[Search Step] SEARCH FOR THE BEST PARAMETER VALUES")
        logger.info(" ".join(['>>', 'Recommender:', self.experimental_settings['recommender']]))
        logger.info(" ".join(['>>', 'Dataset:', self.experimental_settings['dataset']]))
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.CLUSTERING:
            self.starting_cluster()
        else:
            self.starting_recommender()

    def starting_cluster(self):
        """
        TODO: Docstring
        """

        # Starting the counter
        self.start_count()

        # Executing the Random Search
        # search_instance = UnsupervisedLearning(
        #     cluster=self.experimental_settings['cluster'],
        #     distribution=self.experimental_settings['distribution'],
        #     dataset=self.experimental_settings['dataset']
        # )
        # search_instance.fit()

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_search_time(
            data=self.clock_data(),
            dataset=self.experimental_settings['dataset'], algorithm=self.experimental_settings['recommender']
        )

    def starting_recommender(self):
        """
        TODO: Docstring
        """

        # Starting the counter
        self.start_count()

        # Executing the Random Search
        search_instance = SurpriseSearch(
            recommender=self.experimental_settings['recommender'],
            dataset=self.experimental_settings['dataset'])
        search_instance.fit()

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_search_time(
            data=self.clock_data(),
            dataset=self.experimental_settings['dataset'], algorithm=self.experimental_settings['recommender']
        )


if __name__ == '__main__':
    """
    It starts the parameter search
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep2()
    step.read_the_entries()
    step.set_the_logfile()
    step.print_basic_info()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
