from datasets.movielens_one_million import MovielensOneMillion
from datasets.utils.base import Dataset
from datasets.yahoo_movies import YahooMovies


class RegisteredDataset:
    """
    All datasets allowed to the system use, are registered in this class.
    """
    # Registered datasets
    MOVIELENS_ONE_MILLION_DATASET = MovielensOneMillion.system_name
    YAHOO_MOVIES_DATASET = YahooMovies.system_name

    # Allowed to be accessed
    DATASET_LIST = [MOVIELENS_ONE_MILLION_DATASET, YAHOO_MOVIES_DATASET]

    # Default dataset
    DEFAULT_DATASET = MOVIELENS_ONE_MILLION_DATASET

    @staticmethod
    def load_dataset(dataset: str) -> Dataset:
        """
        Load the dataset instance, if the dataset is registered.
        :param dataset: A string with the system name of the dataset.
        :return: A dataset class instance.
        """
        # Movielens One Million
        if dataset == RegisteredDataset.MOVIELENS_ONE_MILLION_DATASET:
            return MovielensOneMillion()
        # Yahoo Movies
        elif dataset == RegisteredDataset.YAHOO_MOVIES_DATASET:
            return YahooMovies()
        else:
            raise "The requested dataset is not registered in the system"

    @staticmethod
    def preprocessing(dataset: str, n_trials: int, n_folds: int):
        """
        Preprocessing the requested dataset, if the dataset is registered.
        :param dataset: A string with the system name of the dataset.
        :param n_trials: An int that represents a number of the experimental trial.
        :param n_folds: An int that represents a number of the k folds.
        """
        # Movielens One Million
        if dataset == RegisteredDataset.MOVIELENS_ONE_MILLION_DATASET:
            instance = MovielensOneMillion()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        # Yahoo Movies
        elif dataset == RegisteredDataset.YAHOO_MOVIES_DATASET:
            instance = YahooMovies()
            instance.mining_data_and_create_fold(n_trials=n_trials, n_folds=n_folds)
        else:
            raise "The requested dataset is not registered in the system"
