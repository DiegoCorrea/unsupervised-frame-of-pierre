import itertools
import logging

from joblib import Parallel, delayed

from evaluations.evaluation_interface import execution_time_fold, applying_map, \
    applying_mrr, applying_mace, applying_mrmc
from evaluations.conformity_algorithms import ConformityAlgorithms
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep5(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step5()
        print(self.experimental_settings)

    @staticmethod
    def set_the_logfile_by_instance(
            dataset: str, recommender: str, trial: int, fold: int, tradeoff: str, distribution: str,
            fairness: str, relevance: str, tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        # Setup Log configuration
        setup_logging(
            log_error="error.log", log_info="info.log",
            save_path=PathDirFile.set_log_metrics_path(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold, tradeoff=tradeoff,
                distribution=distribution, fairness=fairness, relevance=relevance, tradeoff_weight=tradeoff_weight,
                select_item=select_item
            )
        )

    def print_basic_info_by_instance(self, **kwargs):
        """
        TODO: Docstring
        """

        logger.info("$" * 50)
        logger.info("$" * 50)
        # Logging machine data
        self.machine_information()
        logger.info("-" * 50)

        # Logging the experiment setup
        logger.info("[METRIC STEP]")
        logger.info(kwargs)
        logger.info("$" * 50)
        logger.info("$" * 50)

    def main(self):
        """
        TODO: Docstring
        """
        if self.experimental_settings['opt'] == Label.CONFORMITY:
            self.cluster_parallelization()
        elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS:
            self.metrics_parallelization()
        else:
            print(f"Option {self.experimental_settings['opt']} is not registered!")

    @staticmethod
    def applying_evaluation_metrics(
            metrics: list, recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ):
        """
        Function to apply the evaluation metrics.
        """
        load_monitoring = []
        for m in metrics:
            if m == 'MAP':
                load_monitoring.append(
                    applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                 selector)
                )
            elif m == "MRR":
                load_monitoring.append(
                    applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                 selector)
                )
            elif m == "MACE":
                load_monitoring.append(
                    applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                  selector)
                )
            elif m == "MRMC":
                load_monitoring.append(
                    applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                  selector)
                )
            elif m == "TIME":
                execution_time_fold(recommender, dataset, trial, fold,
                                    distribution, fairness, relevance, weight, tradeoff, selector)
            elif m == "RANK":
                load_monitoring.append(
                    applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                 selector)
                )
                load_monitoring.append(
                    applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                 selector)
                )
            elif m == "CALIBRATION":
                load_monitoring.append(
                    applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                  selector)
                )
                load_monitoring.append(
                    applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                  selector)
                )
            else:
                execution_time_fold(recommender, dataset, trial, fold,
                                    distribution, fairness, relevance, weight, tradeoff, selector)
                load_monitoring.append(
                    applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                 selector)
                )
                load_monitoring.append(
                    applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                 selector)
                )
                load_monitoring.append(
                    applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                  selector)
                )
                load_monitoring.append(
                    applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                                  selector)
                )
                break
        return load_monitoring

    def metrics_parallelization(self):
        combination = [
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['trial'], self.experimental_settings['fold'],
            self.experimental_settings['distribution'], self.experimental_settings['fairness'],
            self.experimental_settings['relevance'], self.experimental_settings['weight'],
            self.experimental_settings['tradeoff'], self.experimental_settings['selector']
        ]

        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(self.applying_evaluation_metrics)(
                metrics=self.experimental_settings['metrics'],
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                weight=weight, tradeoff=tradeoff, selector=selector
            ) for recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector
            in list(itertools.product(*combination)))

        # jobs = dict(Counter(load))
        print(load)

    def starting_cluster(
            self, cluster: str, recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str):
        """
        TODO
        """
        if self.experimental_settings["reload"] == "YES":
            try:
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=cluster, metric=Label.JACCARD_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )
                if len(metric_df[Label.JACCARD_SCORE]) > 0:
                    return ""
            except Exception:
                logger.info("Reloading...")
        self.set_the_logfile_by_instance(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector
        )
        self.print_basic_info_by_instance(
                cluster=cluster,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, selector=selector
        )

        # Starting the counter
        self.start_count()

        # Executing the Random Search
        cluster_instance = ConformityAlgorithms(
            cluster=cluster,
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        cluster_instance.prepare_experiment()
        cluster_instance.fit()

        cluster_instance.evaluation()

        # Finishing the counter
        self.finish_count()

        # Saving execution time
        SaveAndLoad.save_conformity_metric_time(
            data=self.clock_data(), cluster=cluster,
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )

    def cluster_parallelization(self):
        """
        TODO
        """
        combination = [
            self.experimental_settings['cluster'],
            self.experimental_settings['recommender'], self.experimental_settings['dataset'],
            self.experimental_settings['trial'], self.experimental_settings['fold'],
            self.experimental_settings['distribution'], self.experimental_settings['fairness'],
            self.experimental_settings['relevance'], self.experimental_settings['weight'],
            self.experimental_settings['tradeoff'], self.experimental_settings['selector']
        ]
        logger.info(f"Total of combinations: {len(combination)}")

        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(self.starting_cluster)(
                cluster=cluster,
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                weight=weight, tradeoff=tradeoff, selector=selector
            ) for cluster, recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector
            in list(itertools.product(*combination)))

        # jobs = dict(Counter(load))
        # print(load)


if __name__ == '__main__':
    """
    It starts the processing step
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep5()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
