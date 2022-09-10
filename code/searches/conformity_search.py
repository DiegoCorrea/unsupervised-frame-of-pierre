import logging

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neural_network import BernoulliRBM
from scikit_pierre.measures.accessible import calibration_measures_funcs

from datasets.registred_datasets import RegisteredDataset
from searches.parameters import ConformityParams
from settings.labels import Label

from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ConformityAlgorithmSearch:
    """
    Class used to lead with the Random Search
    """

    def __init__(self, cluster: str, distribution: str, dataset: str):
        self.measures = ['mae']
        self.dataset = RegisteredDataset.load_dataset(dataset)

        self.distribution_name = distribution
        self.distribution_instance = calibration_measures_funcs(measure=distribution)

        self.algorithm_name = cluster
        self.algorithm_instance = None
        self.params = None
        if cluster == Label.KMEANS:
            self.algorithm_instance = KMeans
            self.params = ConformityParams.KMEANS_SEARCH_PARAMS
        elif cluster == Label.AGGLOMERATIVE:
            self.algorithm_instance = AgglomerativeClustering
            self.params = ConformityParams.AGGLOMERATIVE_CLUSTERING_SEARCH_PARAMS
        elif cluster == Label.BERNOULLI_RBM:
            self.algorithm_instance = BernoulliRBM
            self.params = ConformityParams.BERNOULLI_RBM_SEARCH_PARAMS

    def __search(self):
        """
        Randomized Search Cross Validation to get the best params in the recommender algorithm
        :return: A Random Search instance
        """
        gs = {'best_params': 0}
        return gs

    def fit(self):
        """
        Search and save the best param values
        """
        gs = self.__search()
        # Saving the best
        SaveAndLoad.save_hyperparameters_conformity(
            best_params=gs, dataset=self.dataset.system_name,
            algorithm=self.algorithm_name, distribution=self.distribution_name
        )
