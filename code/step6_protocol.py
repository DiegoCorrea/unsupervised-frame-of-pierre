import itertools
import logging
from statistics import mean, mode

import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame

from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad
from utils.input import Input
from utils.logging_settings import setup_logging
from utils.step import Step

logger = logging.getLogger(__name__)


class PierreStep6(Step):
    """
    TODO: Docstring
    """

    def read_the_entries(self):
        """
        TODO: Docstring
        """
        self.experimental_settings = Input.step6()
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
        # setup_logging(
        #     log_error="error.log", log_info="info.log",
        #     save_path=PathDirFile.set_decision_file(
        #         dataset=dataset, recommender=recommender, trial=trial, fold=fold, tradeoff=tradeoff,
        #         distribution=distribution, fairness=fairness, relevance=relevance, tradeoff_weight=tradeoff_weight,
        #         select_item=select_item
        #     )
        # )
        pass

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
            self.conformity_parallelization()
        # elif self.experimental_settings['opt'] == Label.EVALUATION_METRICS:
        #     self.metrics_parallelization()
        else:
            print(f"Option {self.experimental_settings['opt']} is not registered!")

    # Conformity parallelization
    def load_conformity_metric_jaccard(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.JACCARD_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_cand_items_list.append(metric_df.iloc[0][Label.JACCARD_SCORE])
                users_rec_lists_list.append(metric_df.iloc[1][Label.JACCARD_SCORE])

        merged_metrics_df = DataFrame([
            [mean(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]
        ],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_jaccard_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        jaccard_df = self.load_conformity_metric_jaccard(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        jaccard_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return jaccard_df

    def load_conformity_metric_silhouette(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.SILHOUETTE_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.SILHOUETTE_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.SILHOUETTE_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.SILHOUETTE_SCORE])

        merged_metrics_df = DataFrame([
            [mean(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mean(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_silhouette_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        silhouette_df = self.load_conformity_metric_silhouette(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        silhouette_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return silhouette_df

    def load_conformity_metric_label(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []
        users_cand_items_list = []
        users_rec_lists_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_conformity_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    cluster=conformity, metric=Label.LABEL_SCORE, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(metric_df.iloc[0][Label.LABEL_SCORE])
                users_cand_items_list.append(metric_df.iloc[1][Label.LABEL_SCORE])
                users_rec_lists_list.append(metric_df.iloc[2][Label.LABEL_SCORE])

        merged_metrics_df = DataFrame([
            [mode(users_pref_list), Label.USERS_PREF,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_cand_items_list), Label.USERS_CAND_ITEMS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight],
            [mode(users_rec_lists_list), Label.USERS_REC_LISTS,
             recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.CONFORMITY_DIST_MEANING,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df

    def conformity_labels_metric(
            self, dataset: str, recommender: str, conformity: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        label_df = self.load_conformity_metric_label(
            dataset=dataset, recommender=recommender, conformity=conformity,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, weight=weight, selector=selector
        )
        label_df['COMBINATION'] = "-".join([
            recommender, conformity, tradeoff, distribution, fairness, relevance, selector, weight
        ])

        return label_df

    def conformity_parallelization(self):
        """
        TODO: Docstring
        """
        for dataset in self.experimental_settings['dataset']:
            combination = [
                self.experimental_settings['recommender'], self.experimental_settings['conformity'],
                self.experimental_settings['distribution'], self.experimental_settings['fairness'],
                self.experimental_settings['relevance'], self.experimental_settings['weight'],
                self.experimental_settings['tradeoff'], self.experimental_settings['selector']
            ]

            # Jaccard
            jaccard_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_jaccard_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            jaccard_results = pd.concat(jaccard_output)
            print(jaccard_results)
            SaveAndLoad.save_conformity_metric_compiled(
                data=jaccard_results, dataset=dataset, metric=Label.JACCARD_SCORE
            )

            # Silhouette
            silhouette_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_silhouette_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            silhouette_results = pd.concat(silhouette_output)
            print(silhouette_results)
            SaveAndLoad.save_conformity_metric_compiled(
                data=silhouette_results, dataset=dataset, metric=Label.SILHOUETTE_SCORE
            )

            # Labels
            label_output = Parallel(n_jobs=Constants.N_CORES)(
                delayed(self.conformity_labels_metric)(
                    dataset=dataset, recommender=recommender, conformity=conformity,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                ) for recommender, conformity, distribution, fairness, relevance, weight, tradeoff, selector
                in list(itertools.product(*combination))
            )
            label_results = pd.concat(label_output)
            print(label_results)
            SaveAndLoad.save_conformity_metric_compiled(
                data=label_results, dataset=dataset, metric=Label.LABEL_SCORE
            )

    # Metrics Parallelization

    # def merge_metrics(
    #         self, recommender: str, dataset: str, trial: int, fold: int,
    #         distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str):
    #     """
    #     Merge metrics in a single DataFrame
    #
    #     :param recommender: ...
    #     :param dataset: ...
    #     :param trial: ...
    #     :param fold: ...
    #     :param distribution: ...
    #     :param fairness: ...
    #     :param relevance: ...
    #     :param weight: ...
    #     :param tradeoff: ...
    #     :param selector: ...
    #
    #     :return: ...
    #     """
    #
    #     errors = []
    #     merged_metrics = pd.DataFrame([])
    #
    #     for metric in ['MAP', 'MRR', 'MRMC', 'MACE', 'TIME']:
    #         path = PathDirFile.get_metric_fold_file_by_name(
    #             dataset=dataset, recommender=recommender, trial=trial, fold=fold,
    #             tradeoff=tradeoff, distribution=distribution, fairness=fairness,
    #             relevance=relevance, tradeoff_weight=weight, select_item=selector,
    #             filename=metric + '.csv'
    #         )
    #         try:
    #             metric_df = pd.read_csv(path)
    #             merged_metrics[metric] = metric_df[metric]
    #         except Exception as e:
    #             # logger.error(" - ".join([str(e), "File does not exist or without content that pandas can read!"]))
    #             errors.append("Error")
    #             merged_metrics[metric] = None
    #             continue
    #
    #     try:
    #         merged_metrics.to_csv(
    #             PathDirFile.set_metric_fold_file_by_name(
    #                 recommender=recommender, dataset=dataset, trial=trial, fold=fold,
    #                 distribution=distribution, fairness=fairness, relevance=relevance,
    #                 tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
    #                 filename='ALL_METRICS.csv'
    #             ), index=False
    #         )
    #     except Exception as e:
    #         # logger.error(" - ".join([str(e), "Error due the merged metrics save!"]))
    #         errors.append("Error")
    #
    #     # print(errors)
    #     # print(merged_metrics)
    #     return merged_metrics
    #
    # def load_results(self, recommender: str, dataset: str,
    #                  distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    #                  ) -> DataFrame:
    #     """
    #     Merge metrics in a single DataFrame
    #
    #     :param recommender: ...
    #     :param dataset: ...
    #     :param distribution: ...
    #     :param fairness: ...
    #     :param relevance: ...
    #     :param weight: ...
    #     :param tradeoff: ...
    #     :param selector: ...
    #
    #     :return: ...
    #     """
    #
    #     result_list = []
    #     for trial in range(1, Constants.N_TRIAL_VALUE + 1):
    #         for fold in range(1, Constants.K_FOLDS_VALUE + 1):
    #             result_list.append(merge_metrics(
    #                 dataset=dataset, recommender=recommender, trial=trial, fold=fold,
    #                 tradeoff=tradeoff, distribution=distribution, fairness=fairness,
    #                 relevance=relevance, weight=weight, selector=selector
    #             ))
    #
    #     metrics = pd.concat(result_list)
    #     try:
    #         metrics_av = [metrics[column].sum()
    #                       if column == "TIME" else metrics[column].mean()
    #                       for column in metrics.columns.tolist()]
    #         result = pd.DataFrame([metrics_av], columns=metrics.columns.tolist())
    #     except Exception as e:
    #         print(metrics)
    #     result['COMBINATION'] = "-".join([recommender, tradeoff, distribution, fairness, relevance, selector, weight])
    #     return result
    #
    # def metrics_parallelization(self):
    #     """
    #     TODO: Docstring
    #     """
    #     combination = [
    #         self.experimental_settings['recommender'], self.experimental_settings['dataset'],
    #         self.experimental_settings['distribution'], self.experimental_settings['fairness'],
    #         self.experimental_settings['relevance'], self.experimental_settings['weight'],
    #         self.experimental_settings['tradeoff'], self.experimental_settings['selector']
    #     ]
    #     output = Parallel(n_jobs=Constants.N_CORES)(
    #         delayed(load_results)(
    #             recommender=recommender, dataset=dataset,
    #             distribution=distribution, fairness=fairness, relevance=relevance,
    #             weight=weight, tradeoff=tradeoff, selector=selector
    #         ) for recommender, dataset, distribution, fairness, relevance, weight, tradeoff, selector
    #         in list(itertools.product(*combination))
    #     )
    #     # print(output)
    #     results = pd.concat(output)
    #     results['CMC'] = results['MAP'] / results['MRMC']
    #     results['CCE'] = results['MAP'] / results['MACE']
    #     results['PERFORMANCE'] = results['CCE'] + results['CMC']
    #     results.sort_values(by=['MAP'], ascending=False, inplace=True)
    #     # print(results)
    #     results.to_csv(PathDirFile.set_decision_file(self.experimental_settings['dataset'][0]), index=False)


if __name__ == '__main__':
    """
    Starting the decision protocol
    """
    logger.info(" ".join(['+' * 10, 'System Starting', '+' * 10]))
    step = PierreStep6()
    step.read_the_entries()
    step.main()
    logger.info(" ".join(['+' * 10, 'System shutdown', '+' * 10]))
