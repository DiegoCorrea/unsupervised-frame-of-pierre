import logging

from fcmeans import FCM
from pandas import DataFrame
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS, BisectingKMeans, SpectralClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.neural_network import BernoulliRBM

from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.classes.genre import genre_probability_approach

from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from settings.labels import Label
import pandas as pd

from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ConformityAlgorithms:
    """
    TODO
    """

    def __init__(self, dataset: str, fold: int, trial: int, recommender: str, cluster: str,
                 distribution: str, tradeoff: str, fairness: str, relevance: str, weight: str, selector: str):
        """
        TODO
        """
        self.dataset = RegisteredDataset.load_dataset(dataset)
        self.fold_int = fold
        self.trial_int = trial
        self.recommender_str = recommender
        self.conformity_str = cluster
        self.distribution_str = distribution
        self.tradeoff_str = tradeoff
        self.fairness_str = fairness
        self.relevance_str = relevance
        self.weight_str = weight
        self.selector_str = selector

        self.params = None

        self.users_preferences_labels = None
        self.users_candidate_labels = None
        self.users_recommendation_labels = None

        self.users_preferences_instance = None
        self.users_candidate_instance = None
        self.users_recommendation_instance = None

        self.items_classes_set = genre_probability_approach(item_set=self.dataset.get_items())
        self.dist_func = distributions_funcs_pandas(distribution=self.distribution_str)

    def __load_conformity_algorithm_instance(self):
        """
        It prepares the algorithm instance.
        """
        params = SaveAndLoad.load_hyperparameters_conformity(
            dataset=self.dataset.system_name, recommender=self.recommender_str,
            cluster=self.conformity_str, distribution=self.distribution_str
        )

        # K-Means Variations
        if self.conformity_str == Label.KMEANS:
            self.users_preferences_instance = KMeans(n_clusters=params['n_clusters'], init='k-means++')
            self.users_candidate_instance = KMeans(n_clusters=params['n_clusters'], init='k-means++')
            self.users_recommendation_instance = KMeans(n_clusters=params['n_clusters'], init='k-means++')
        elif self.conformity_str == Label.FCM:
            self.users_preferences_instance = FCM(n_clusters=params['n_clusters'])
            self.users_candidate_instance = FCM(n_clusters=params['n_clusters'])
            self.users_recommendation_instance = FCM(n_clusters=params['n_clusters'])
        elif self.conformity_str == Label.BISECTING:
            self.users_preferences_instance = BisectingKMeans(n_clusters=params['n_clusters'], init='k-means++')
            self.users_candidate_instance = BisectingKMeans(n_clusters=params['n_clusters'], init='k-means++')
            self.users_recommendation_instance = BisectingKMeans(n_clusters=params['n_clusters'], init='k-means++')

        # Hierarchical Variations
        elif self.conformity_str == Label.AGGLOMERATIVE:
            self.users_preferences_instance = AgglomerativeClustering(n_clusters=params['n_clusters'])
            self.users_candidate_instance = AgglomerativeClustering(n_clusters=params['n_clusters'])
            self.users_recommendation_instance = AgglomerativeClustering(n_clusters=params['n_clusters'])

        # Spectral Variations
        elif self.conformity_str == Label.SPECTRAL:
            self.users_preferences_instance = SpectralClustering(n_clusters=params['n_clusters'])
            self.users_candidate_instance = SpectralClustering(n_clusters=params['n_clusters'])
            self.users_recommendation_instance = SpectralClustering(n_clusters=params['n_clusters'])

        # Tree Variations
        elif self.conformity_str == Label.BIRCH:
            self.users_preferences_instance = Birch(n_clusters=params['n_clusters'])
            self.users_candidate_instance = Birch(n_clusters=params['n_clusters'])
            self.users_recommendation_instance = Birch(n_clusters=params['n_clusters'])
        elif self.conformity_str == Label.IF:
            self.users_preferences_instance = IsolationForest()
            self.users_candidate_instance = IsolationForest()
            self.users_recommendation_instance = IsolationForest()

        # Search Variations
        elif self.conformity_str == Label.DBSCAN:
            self.users_preferences_instance = DBSCAN(
                min_samples=params['min_samples'], eps=params['eps'], metric=params['metric']
            )
            self.users_candidate_instance = DBSCAN(
                min_samples=params['min_samples'], eps=params['eps'], metric=params['metric']
            )
            self.users_recommendation_instance = DBSCAN(
                min_samples=params['min_samples'], eps=params['eps'], metric=params['metric']
            )
        elif self.conformity_str == Label.OPTICS:
            self.users_preferences_instance = OPTICS(
                min_samples=params['min_samples'], eps=params['eps'], metric=params['metric']
            )
            self.users_candidate_instance = OPTICS(
                min_samples=params['min_samples'], eps=params['eps'], metric=params['metric']
            )
            self.users_recommendation_instance = OPTICS(
                min_samples=params['min_samples'], eps=params['eps'], metric=params['metric']
            )

    def __load_users_preference_distribution(self):
        """
        TODO
        """
        self.distribution_instance = calibration_measures_funcs(measure=self.distribution_str)
        self.users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
            dataset=self.dataset.system_name, trial=self.trial_int, fold=self.fold_int,
            distribution=self.distribution_str
        )

    def __load_users_candidate_items_distribution(self):
        """
        TODO
        """
        self.users_candidate_items = SaveAndLoad.load_candidate_items(
            dataset=self.dataset.get_dataset_name(), fold=self.fold_int, trial=self.trial_int,
            algorithm=self.recommender_str
        )
        self.users_cand_items_dist_df = pd.concat([
            self.dist_func(
                user_pref_set=self.users_candidate_items[
                    self.users_candidate_items['USER_ID'] == user_id
                ].sort_values(by=Label.TRANSACTION_VALUE).head(Constants.RECOMMENDATION_LIST_SIZE),
                item_classes_set=self.items_classes_set
            ) for user_id in self.users_candidate_items['USER_ID'].unique().tolist()
        ])

    def __load_users_recommendation_lists_distribution(self):
        """
        TODO
        """
        self.users_recommendation_lists = SaveAndLoad.load_recommendation_lists(
            dataset=self.dataset.get_dataset_name(), recommender=self.recommender_str, trial=self.trial_int, fold=self.fold_int,
            tradeoff=self.tradeoff_str, distribution=self.distribution_str, fairness=self.fairness_str,
            relevance=self.relevance_str, tradeoff_weight=self.weight_str, select_item=self.selector_str

        )
        self.users_rec_lists_dist_df = pd.concat([
            self.dist_func(
                user_pref_set=self.users_recommendation_lists[self.users_recommendation_lists['USER_ID'] == user_id],
                item_classes_set=self.items_classes_set
            ) for user_id in self.users_recommendation_lists['USER_ID'].unique().tolist()
        ])

    def prepare_experiment(self):
        """
        TODO
        """
        self.__load_conformity_algorithm_instance()
        self.__load_users_preference_distribution()
        self.__load_users_candidate_items_distribution()
        self.__load_users_recommendation_lists_distribution()

    def fit(self):
        """
        It trains and find the clusters in the data.
        """
        # Train
        if self.conformity_str != Label.FCM:
            self.users_preferences_instance = self.users_preferences_instance.fit(
                X=self.users_pref_dist_df
            )
            self.users_candidate_instance = self.users_candidate_instance.fit(
                X=self.users_cand_items_dist_df
            )
            self.users_recommendation_instance = self.users_recommendation_instance.fit(
                X=self.users_rec_lists_dist_df
            )
        else:
            self.users_preferences_instance.fit(
                X=self.users_pref_dist_df.to_numpy()
            )
            self.users_candidate_instance.fit(
                X=self.users_cand_items_dist_df.to_numpy()
            )
            self.users_recommendation_instance.fit(
                X=self.users_rec_lists_dist_df.to_numpy()
            )

        if self.conformity_str == Label.KMEANS or self.conformity_str == Label.BISECTING:
            self.users_preferences_labels = self.users_preferences_instance.predict(
                X=self.users_pref_dist_df
            )
            self.users_candidate_labels = self.users_candidate_instance.predict(
                X=self.users_cand_items_dist_df
            )
            self.users_recommendation_labels = self.users_recommendation_instance.predict(
                X=self.users_rec_lists_dist_df
            )
        elif self.conformity_str == Label.AGGLOMERATIVE or self.conformity_str == Label.IF or \
                self.conformity_str == Label.BIRCH or self.conformity_str == Label.OPTICS or \
                self.conformity_str == Label.SPECTRAL or self.conformity_str == Label.DBSCAN:
            self.users_preferences_labels = self.users_preferences_instance.fit_predict(
                X=self.users_pref_dist_df
            )
            self.users_candidate_labels = self.users_candidate_instance.fit_predict(
                X=self.users_cand_items_dist_df
            )
            self.users_recommendation_labels = self.users_recommendation_instance.fit_predict(
                X=self.users_rec_lists_dist_df
            )
        elif self.conformity_str == Label.FCM:
            self.users_preferences_labels = self.users_preferences_instance.predict(
                X=self.users_pref_dist_df.to_numpy()
            )
            self.users_candidate_labels = self.users_candidate_instance.predict(
                X=self.users_cand_items_dist_df.to_numpy()
            )
            self.users_recommendation_labels = self.users_recommendation_instance.predict(
                X=self.users_rec_lists_dist_df.to_numpy()
            )

    def __silhouette_avg(self):
        print(self.users_preferences_labels)
        print(set(self.users_preferences_labels))
        user_pref_silhouette_avg = 0.0
        users_cand_items_silhouette_avg = 0.0
        users_rec_lists_silhouette_avg = 0.0

        if len(set(self.users_preferences_labels)) > 1:
            user_pref_silhouette_avg = silhouette_score(self.users_pref_dist_df, self.users_preferences_labels)

        if len(set(self.users_candidate_labels)) > 1:
            users_cand_items_silhouette_avg = silhouette_score(self.users_cand_items_dist_df, self.users_candidate_labels)

        if len(set(self.users_recommendation_labels)) > 1:
            users_rec_lists_silhouette_avg = silhouette_score(self.users_rec_lists_dist_df, self.users_recommendation_labels)

        data = DataFrame(
            [[user_pref_silhouette_avg], [users_cand_items_silhouette_avg], [users_rec_lists_silhouette_avg]],
            columns=[Label.SILHOUETTE_SCORE], index=[Label.USERS_PREF, Label.USERS_CAND_ITEMS, Label.USERS_REC_LISTS]
        )

        print("Silhouette avg:", data)

        SaveAndLoad.save_conformity_metric(
            data=data, metric=Label.SILHOUETTE_SCORE, cluster=self.conformity_str, recommender=self.recommender_str,
            dataset=self.dataset.get_dataset_name(), trial=self.trial_int, fold=self.fold_int,
            distribution=self.distribution_str, fairness=self.fairness_str, relevance=self.relevance_str,
            weight=self.weight_str, tradeoff=self.tradeoff_str, selector=self.selector_str
        )

    def __group_jaccard_score(self):
        users_cand_item_score_float = jaccard_score(
            self.users_preferences_labels, self.users_candidate_labels, average='macro'
        )
        user_rec_lists_score_float = jaccard_score(
            self.users_preferences_labels, self.users_recommendation_labels, average='macro'
        )

        data = DataFrame(
            [[users_cand_item_score_float], [user_rec_lists_score_float]],
            columns=[Label.JACCARD_SCORE], index=[Label.USERS_CAND_ITEMS, Label.USERS_REC_LISTS]
        )

        print("JACCARD SCORE:", data)

        SaveAndLoad.save_conformity_metric(
            data=data, metric=Label.JACCARD_SCORE, cluster=self.conformity_str, recommender=self.recommender_str,
            dataset=self.dataset.get_dataset_name(), trial=self.trial_int, fold=self.fold_int,
            distribution=self.distribution_str, fairness=self.fairness_str, relevance=self.relevance_str,
            weight=self.weight_str, tradeoff=self.tradeoff_str, selector=self.selector_str
        )

    def evaluation(self):
        self.__silhouette_avg()
        self.__group_jaccard_score()
