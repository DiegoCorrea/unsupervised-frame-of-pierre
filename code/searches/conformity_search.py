import itertools
import logging

from fcmeans import FCM
from joblib import Parallel, delayed
from numpy import mean
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, OPTICS, Birch, AgglomerativeClustering, KMeans, SpectralClustering, BisectingKMeans
from sklearn.model_selection import ParameterGrid

from scikit_pierre.measures.accessible import calibration_measures_funcs
from sklearn.metrics import silhouette_score

from datasets.registred_datasets import RegisteredDataset
from searches.parameters import ConformityParams
from settings.constants import Constants
from settings.labels import Label
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ManualConformityAlgorithmSearch:
    """
    Class used to lead with the Random Search
    """

    def __init__(self, experimental_settings: dict):
        self.experimental_settings = experimental_settings
        self.dataset = RegisteredDataset.load_dataset(self.experimental_settings['dataset'])

        self.distribution_name = self.experimental_settings['distribution']
        self.distribution_instance = calibration_measures_funcs(measure=self.experimental_settings['distribution'])

        self.param_grid = ConformityParams.CLUSTER_PARAMS_GRID

        self.cluster_params = ConformityParams.CLUSTER_PARAMS

    @staticmethod
    def load_conformity_algorithm_instance(conformity_str, params):
        """
        TODO
        """

        # K-Means Variations
        if conformity_str == Label.KMEANS:
            temp = KMeans(n_clusters=params['n_clusters'], init='k-means++')
            return temp
        elif conformity_str == Label.FCM:
            return FCM(n_clusters=params['n_clusters'])
        elif conformity_str == Label.BISECTING:
            return BisectingKMeans(n_clusters=params['n_clusters'], init='k-means++')

        # Hierarchical Variations
        elif conformity_str == Label.AGGLOMERATIVE:
            return AgglomerativeClustering(n_clusters=params['n_clusters'])

        # Spectral Variations
        elif conformity_str == Label.SPECTRAL:
            return SpectralClustering(n_clusters=params['n_clusters'])

        # Tree Variations
        elif conformity_str == Label.BIRCH:
            return Birch(n_clusters=params['n_clusters'])
        elif conformity_str == Label.IF:
            return IsolationForest()

        # Search Variations
        elif conformity_str == Label.DBSCAN:
            return DBSCAN(min_samples=params['min_samples'], eps=params['eps'], metric=params['metric'])
        elif conformity_str == Label.OPTICS:
            return OPTICS(min_samples=params['min_samples'], eps=params['eps'], metric=params['metric'])

    @staticmethod
    def fit(conformity_str, users_pref_dist_df, users_preferences_instance):
        """
        TODO
        """
        # Train
        if conformity_str != Label.FCM:
            users_preferences_instance = users_preferences_instance.fit(
                X=users_pref_dist_df
            )
        else:
            users_preferences_instance.fit(
                X=users_pref_dist_df.to_numpy()
            )

        # Clustering
        if conformity_str == Label.KMEANS or conformity_str == Label.BISECTING:
            return users_preferences_instance.predict(
                X=users_pref_dist_df
            )
        elif conformity_str == Label.AGGLOMERATIVE or conformity_str == Label.IF or \
                conformity_str == Label.BIRCH or conformity_str == Label.OPTICS or conformity_str == Label.SPECTRAL or \
                conformity_str == Label.DBSCAN:
            return users_preferences_instance.fit_predict(
                X=users_pref_dist_df
            )
        elif conformity_str == Label.FCM:
            return users_preferences_instance.predict(
                X=users_pref_dist_df.to_numpy()
            )

    def search(self, params, conformity_str):
        silhouette_list = []

        for trial in range(1, Constants.N_TRIAL_VALUE + 1):
            for fold in range(1, Constants.K_FOLDS_VALUE + 1):
                # Load users' preferences distributions
                users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
                    dataset=self.dataset.system_name, trial=trial, fold=fold,
                    distribution=self.distribution_name
                )
                users_preferences_instance = ManualConformityAlgorithmSearch.load_conformity_algorithm_instance(
                    conformity_str=conformity_str, params=params
                )
                clusters = ManualConformityAlgorithmSearch.fit(
                    conformity_str=conformity_str, users_pref_dist_df=users_pref_dist_df,
                    users_preferences_instance=users_preferences_instance
                )

                if len(set(clusters)) == 1:
                    continue

                silhouette_list.append(silhouette_score(users_pref_dist_df, clusters))
        return {
            "silhouette": mean(silhouette_list) if len(silhouette_list) else 0,
            "params": params
        }

    def run(self, conformity_str: str):
        """
        Start to run the Manual Grid Search for Unsupervised Learning Clustering Algorithms
        """
        best_silhouette = 0
        best_param = None

        # Chosen the parameter structure
        if conformity_str in Label.CLUSTERING_ALGORITHMS:
            params_list = self.cluster_params
        else:
            params_list = self.param_grid

        # Performing manual gridsearch

        payload = Parallel(n_jobs=Constants.N_CORES, verbose=10)(
            delayed(self.search)(
                params=params, conformity_str=conformity_str
            ) for params in list(ParameterGrid(params_list)))

        for params in payload:
            if abs(params["silhouette"]) > abs(best_silhouette):
                best_silhouette = abs(params["silhouette"])
                best_param = params["params"]

        # Saving the best
        SaveAndLoad.save_hyperparameters_conformity(
            best_params=best_param, dataset=self.dataset.system_name,
            algorithm=conformity_str, distribution=self.distribution_name
        )
        # return {
        #     "silhouette": best_silhouette,
        #     "best_params": best_param
        # }

