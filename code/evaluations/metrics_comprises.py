from statistics import mean

from pandas import DataFrame

from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad


class MetricComprises:
    @staticmethod
    def it_comprises_recommender_metric(
            dataset: str, recommender: str, metric: str,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ) -> DataFrame:
        """
        TODO: Docstring
        """

        users_pref_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                metric_df = SaveAndLoad.load_recommender_metric(
                    dataset=dataset, trial=trial, fold=fold,
                    metric=metric, recommender=recommender,
                    distribution=distribution, fairness=fairness, relevance=relevance,
                    weight=weight, tradeoff=tradeoff, selector=selector
                )

                users_pref_list.append(mean(metric_df[metric].tolist()))

        merged_metrics_df = DataFrame([
            [mean(users_pref_list), metric,
             recommender, metric, tradeoff, distribution, fairness, relevance, selector, weight]],
            columns=[
                Label.EVALUATION_METRICS, Label.RECOMMENDER_METRIC,
                Label.RECOMMENDER, Label.CONFORMITY, Label.TRADEOFF, Label.DISTRIBUTION_LABEL,
                Label.CALIBRATION_MEASURE_LABEL, Label.RELEVANCE, Label.SELECTOR_LABEL, Label.TRADEOFF_WEIGHT_LABEL
            ]
        )

        return merged_metrics_df
