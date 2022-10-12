import json

from pandas import DataFrame, read_csv

from settings.labels import Label
from settings.path_dir_file import PathDirFile


class SaveAndLoad:
    """
    TODO: Docstring
    """

    # ########################################################################################### #
    # [STEP 2] Search step methods - Best Parameters
    # ########################################################################################### #
    @staticmethod
    def save_hyperparameters_recommender(best_params: dict, dataset: str, algorithm: str):
        """
        TODO: Docstring
        """
        with open(PathDirFile.set_recommender_hyperparameter_file(
                opt=Label.RECOMMENDER, dataset=dataset, algorithm=algorithm), 'w') as fp:
            json.dump(best_params['mae'], fp)

    @staticmethod
    def load_hyperparameters_recommender(dataset: str, algorithm: str):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_recommender_hyperparameter_file(
            opt=Label.RECOMMENDER, dataset=dataset, algorithm=algorithm)
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    @staticmethod
    def save_hyperparameters_conformity(best_params: dict, dataset: str, algorithm: str, distribution: str):
        """
        TODO: Docstring
        """
        with open(PathDirFile.set_conformity_hyperparameter_file(
                opt=Label.CONFORMITY, dataset=dataset, algorithm=algorithm, distribution=distribution), 'w') as fp:
            json.dump(best_params['mae'], fp)

    @staticmethod
    def load_hyperparameters_conformity(dataset: str, algorithm: str, distribution: str):
        """
        TODO: Docstring
        """
        path_to_open = PathDirFile.get_conformity_hyperparameter_file(
            opt=Label.CONFORMITY, dataset=dataset, algorithm=algorithm, distribution=distribution)
        with open(path_to_open) as json_file:
            params = json.load(json_file)

        return params

    # ########################################################################################### #
    # [STEP 2] Search step methods - Time
    # ########################################################################################### #
    @staticmethod
    def save_search_time(data: DataFrame, dataset: str, algorithm: str):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_search_time_file(dataset=dataset, algorithm=algorithm),
            index=False
        )

    # ########################################################################################### #
    # [STEP 3] Processing step methods - Candidate Items
    # ########################################################################################### #
    @staticmethod
    def save_candidate_items(data: DataFrame, dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_candidate_items_file(
                dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
            ),
            index=False
        )

    @staticmethod
    def load_candidate_items(dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        candidate_items_path = PathDirFile.get_candidate_items_file(
            dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
        )
        return read_csv(candidate_items_path)

    # ########################################################################################### #
    # [STEP 3] Processing step methods - Time
    # ########################################################################################### #
    @staticmethod
    def save_processing_time(data: DataFrame, dataset: str, algorithm: str, trial: int, fold: int):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_processing_time_file(
                dataset=dataset, algorithm=algorithm, fold=fold, trial=trial
            ),
            index=False
        )

    # ########################################################################################### #
    # [STEP 3] Post-Processing step methods - Recommendation Lists
    # ########################################################################################### #
    @staticmethod
    def load_recommendation_lists(
        dataset: str, recommender: str, trial: int, fold: int,
        tradeoff: str, distribution: str, fairness: str, relevance: str,
        tradeoff_weight: str, select_item: str
    ):
        """
        TODO: Docstring
        """
        recommendation_list_path = PathDirFile.get_recommendation_list_file(
            dataset=dataset, recommender=recommender, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, tradeoff_weight=tradeoff_weight, select_item=select_item
        )
        return read_csv(recommendation_list_path)

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Time
    # ########################################################################################### #
    @staticmethod
    def save_conformity_metric_time(
            data: DataFrame,
            cluster: str, recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_conformity_metrics_time_file(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector, cluster=cluster
            ),
            index=False
        )

    # ########################################################################################### #
    # [STEP 5] Metrics step methods - Conformity Evaluation
    # ########################################################################################### #
    @staticmethod
    def save_conformity_metric(
        data: DataFrame,
        cluster: str, metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ):
        """
        TODO: Docstring
        """
        data.to_csv(
            PathDirFile.set_conformity_metric_fold_file_by_name(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
                cluster=cluster, filename=metric + '.csv'
            )
        )

    @staticmethod
    def load_conformity_metric(
        cluster: str, metric: str, recommender: str, dataset: str, trial: int, fold: int,
        distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str,
        ext: str = 'csv'
    ) -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.get_conformity_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            cluster=cluster, filename=metric + '.' + ext
        )
        return read_csv(path)

    # ########################################################################################### #
    # [STEP 6] Compile Metrics step methods - Conformity Evaluation
    # ########################################################################################### #
    @staticmethod
    def save_conformity_metric_compiled(data: DataFrame, dataset: str, metric: str, ext: str = 'csv'):
        """
        TODO: Docstring
        """
        path = PathDirFile.set_conformity_metric_file(
            dataset=dataset, filename=metric, ext=ext
        )
        data.to_csv(path)

    @staticmethod
    def load_conformity_metric_compiled(dataset: str, metric: str, ext: str = 'csv') -> DataFrame:
        """
        TODO: Docstring
        """
        path = PathDirFile.set_conformity_metric_file(
            dataset=dataset, filename=metric, ext=ext
        )
        return read_csv(path)
