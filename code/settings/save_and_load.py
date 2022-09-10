import json

from pandas import DataFrame

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