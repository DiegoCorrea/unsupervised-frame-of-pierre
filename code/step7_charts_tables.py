import logging

from graphics.conformity import ConformityGraphics
# from evaluations.best_worst import best_and_worst_systems, best_and_worst_fairness_measure
# from evaluations.hypothesis import welch
# from graphics.research_questions.perspectives import components_box_graphic, fairness_box_graphic
from settings.labels import Label

from settings.save_and_load import SaveAndLoad

from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep7(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step7()
        logger.info(self.experimental_settings)

    @staticmethod
    def set_the_logfile_by_instance(
            dataset: str, recommender: str, trial: int, fold: int, tradeoff: str, distribution: str,
            fairness: str, relevance: str, tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        # setup_logging(
        #     log_error="error.log", log_info="info.log",
        #     save_path=PathDirFile.set_decision_file(
        #         dataset=dataset, recommender=recommender, trial=trial, fold=fold, tradeoff=tradeoff,
        #         distribution=distribution, fairness=fairness, relevance=relevance, tradeoff_weight=tradeoff_weight,
        #         select_item=select_item
        #     )
        # )
        pass

    def __init__(self):
        """
        TODO: Docstring
        """

        super().__init__()
        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging the experiment setup
        logger.info("[CHARTS | TABLES | TESTS]")
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """

        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.conformity()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS:
            self.recommender()
        else:
            pass

    def conformity(self):
        if self.experimental_settings['view'] == Label.DATASET_CHART:
            self.conformity_charts()
        elif self.experimental_settings['view'] == Label.DATASET_ANALYZE:
            self.conformity_analyses()
        else:
            pass

    def conformity_charts(self):
        for dataset_name in self.experimental_settings['dataset']:
            print("|" * 100)
            print("-"*10, " ", dataset_name, " ", "-"*10)
            print("|" * 100)

            jaccard_results = SaveAndLoad.load_conformity_metric_compiled(
                dataset=dataset_name, metric=Label.JACCARD_SCORE
            )
            ConformityGraphics.silhouette_group_bar(
                jaccard_results, dataset_name, Label.REGISETRED_UNSUPERVISED, Label.JACCARD_SCORE
            )

            silhlouete_results = SaveAndLoad.load_conformity_metric_compiled(
                dataset=dataset_name, metric=Label.SILHOUETTE_SCORE
            )
            ConformityGraphics.silhouette_group_bar(
                silhlouete_results, dataset_name, Label.REGISETRED_UNSUPERVISED, Label.SILHOUETTE_SCORE
            )

    def conformity_analyses(self):
        for dataset_name in self.experimental_settings['dataset']:
            print("|" * 100)
            print("-"*10, " ", dataset_name, " ", "-"*10)
            print("|" * 100)

            jaccard_results = SaveAndLoad.load_conformity_metric_compiled(
                dataset=dataset_name, metric=Label.JACCARD_SCORE
            )
            print(jaccard_results)

            silhlouete_results = SaveAndLoad.load_conformity_metric_compiled(
                dataset=dataset_name, metric=Label.SILHOUETTE_SCORE
            )
            print(silhlouete_results)

    def recommender(self):
        pass
    #     if setup_config['opt'] == "CHART":
    #         charts(setup_config)
    #     elif setup_config['opt'] == "ANALYZE":
    #         analyses(setup_config)
    #     else:
    #         pass
    #
    # def charts(self):
    #     for dataset_name in setup_config['dataset']:
    #         data = pd.read_csv(PathDirFile.get_decision_file(dataset_name))
    #
    #         # Ranking
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MAP")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MAP")
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRR")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRR")
    #
    #         # Calibration
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MACE")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MACE")
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRMC")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRMC")
    #
    #         # Coefficients
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CCE")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CCE")
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CMC")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CMC")
    #
    #         # Coefficients
    #         components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="PERFORMANCE")
    #         fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="PERFORMANCE")
    #
    # def analyses(self):
    #     for dataset_name in setup_config['dataset']:
    #         print("|" * 100)
    #         print("-"*10, " ", dataset_name, " ", "-"*10)
    #         print("|" * 100)
    #
    #         results = pd.read_csv(PathDirFile.get_decision_file(dataset_name))
    #         results.fillna(0.0)
    #
    #         # execution_time_analyze(data=results)
    #         best_and_worst_systems(data=results, order_by_metric='MAP', ascending=False)
    #         best_and_worst_systems(data=results, order_by_metric='MRR', ascending=False)
    #         best_and_worst_systems(data=results, order_by_metric='MACE', ascending=True)
    #         best_and_worst_systems(data=results, order_by_metric='CCE', ascending=True)
    #
    #         best_and_worst_fairness_measure(data=results, order_by_metric='MAP', ascending=False)
    #         best_and_worst_fairness_measure(data=results, order_by_metric='MRR', ascending=False)
    #         best_and_worst_fairness_measure(data=results, order_by_metric='MACE', ascending=True)
    #         best_and_worst_fairness_measure(data=results, order_by_metric='CCE', ascending=True)
    #
    #         welch(data=results, order_by_metric='MAP', ascending=False)
    #         welch(data=results, order_by_metric='MRR', ascending=False)
    #         welch(data=results, order_by_metric='MACE', ascending=True)
    #         welch(data=results, order_by_metric='CCE', ascending=True)
    #
    #         print("|" * 100)


if __name__ == '__main__':
    """
    Starting the decision protocol
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep7()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
