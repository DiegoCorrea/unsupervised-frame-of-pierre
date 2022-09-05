import json
import logging

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.neural_network import BernoulliRBM

from processing.conversions.pandas_surprise import PandasSurprise
from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from surprise_params import SurpriseParams
from settings.labels import Label

from settings.path_dir_file import PathDirFile

logger = logging.getLogger(__name__)


class UnsupervisedLearning:
    """
    Class used to lead with the Random Search
    """

    def __init__(self, cluster: str, distribution: str, dataset: str):
        self.measures = ['mae']
        self.dataset = RegisteredDataset.load_dataset(dataset)
        self.algorithm_name = cluster
        self.algorithm_instance = None
        self.params = None
        if cluster == Label.KMEANS:
            self.algorithm_instance = KMeans
            self.params = SurpriseParams.SVD_SEARCH_PARAMS
        elif cluster == Label.NMF:
            self.algorithm_instance = NMF
            self.params = SurpriseParams.NMF_SEARCH_PARAMS
        else:
            self.algorithm_instance = BernoulliRBM
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
        # Saving the the best
        with open(PathDirFile.set_hyperparameter_file(self.dataset.system_name, self.recommender_name), 'w') as fp:
            json.dump(gs.best_params['mae'], fp)
