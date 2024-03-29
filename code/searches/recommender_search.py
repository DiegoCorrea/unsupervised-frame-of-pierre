import logging

from surprise import SVD, KNNBasic
from surprise.model_selection import RandomizedSearchCV
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise.prediction_algorithms.matrix_factorization import SVDpp, NMF

from processing.conversions.pandas_surprise import PandasSurprise
from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from settings.save_and_load import SaveAndLoad
from searches.parameters import SurpriseParams
from settings.labels import Label

logger = logging.getLogger(__name__)


class RecommenderSearch:
    """
    Class used to lead with the Random Search
    """

    def __init__(self, recommender: str, dataset: str):
        self.measures = ['rmse', 'mae']
        self.dataset = RegisteredDataset.load_dataset(dataset)
        self.recommender_name = recommender
        self.recommender = None
        self.params = None
        if recommender == Label.SVD:
            self.recommender = SVD
            self.params = SurpriseParams.SVD_SEARCH_PARAMS
        elif recommender == Label.NMF:
            self.recommender = NMF
            self.params = SurpriseParams.NMF_SEARCH_PARAMS
        elif recommender == Label.CO_CLUSTERING:
            self.recommender = CoClustering
            self.params = SurpriseParams.CLUSTERING_SEARCH_PARAMS
        elif recommender == Label.ITEM_KNN_BASIC:
            self.recommender = KNNBasic
            self.params = SurpriseParams.ITEM_KNN_SEARCH_PARAMS
        elif recommender == Label.USER_KNN_BASIC:
            self.recommender = KNNBasic
            self.params = SurpriseParams.USER_KNN_SEARCH_PARAMS
        else:
            self.recommender = SVDpp
            self.params = SurpriseParams.SVDpp_SEARCH_PARAMS

    def __search(self):
        """
        Randomized Search Cross Validation to get the best params in the recommender algorithm
        :return: A Random Search instance
        """
        gs = RandomizedSearchCV(algo_class=self.recommender, param_distributions=self.params, measures=self.measures,
                                n_iter=Constants.N_INTER, cv=Constants.K_FOLDS_VALUE,
                                n_jobs=Constants.N_CORES, joblib_verbose=100, random_state=42)
        gs.fit(PandasSurprise.pandas_transform_all_dataset_to_surprise(self.dataset.get_transactions()))
        return gs

    def fit(self):
        """
        Search and save the best param values
        """
        gs = self.__search()
        # Saving
        SaveAndLoad.save_hyperparameters_recommender(
            best_params=gs.best_params, dataset=self.dataset.system_name, algorithm=self.recommender_name
        )
