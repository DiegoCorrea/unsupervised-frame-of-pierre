import logging

from pandas import DataFrame
from sklearn.cluster import KMeans, AgglomerativeClustering
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

        self.users_preferences_instance = None
        self.users_candidate_instance = None
        self.users_recommendation_instance = None

        self.items_classes_set = genre_probability_approach(item_set=self.dataset.get_items())
        self.dist_func = distributions_funcs_pandas(distribution=self.distribution_str)

    def __load_conformity_algorithm_instance(self):
        if self.conformity_str == Label.KMEANS:
            self.users_preferences_instance = KMeans(n_clusters=3)
            self.users_candidate_instance = KMeans(n_clusters=3)
            self.users_recommendation_instance = KMeans(n_clusters=3)
        elif self.conformity_str == Label.AGGLOMERATIVE:
            self.users_preferences_instance = AgglomerativeClustering(n_clusters=3)
            self.users_candidate_instance = AgglomerativeClustering(n_clusters=3)
            self.users_recommendation_instance = AgglomerativeClustering(n_clusters=3)
        elif self.conformity_str == Label.BERNOULLI_RBM:
            self.users_preferences_instance = BernoulliRBM(n_components=3)
            self.users_candidate_instance = BernoulliRBM(n_components=3)
            self.users_recommendation_instance = BernoulliRBM(n_components=3)

    def __load_users_preference_distribution(self):
        users_preference_set = self.dataset.get_train_transactions(trial=self.trial_int, fold=self.fold_int)
        self.distribution_instance = calibration_measures_funcs(measure=self.distribution_str)
        self.users_pref_dist_df = pd.concat([
            self.dist_func(
                user_pref_set=users_preference_set[users_preference_set['USER_ID'] == user_id],
                item_classes_set=self.items_classes_set
            ) for user_id in users_preference_set['USER_ID'].unique().tolist()
        ])

    def __load_users_candidate_items_distribution(self):
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
        self.__load_conformity_algorithm_instance()
        self.__load_users_preference_distribution()
        self.__load_users_candidate_items_distribution()
        self.__load_users_recommendation_lists_distribution()

    def fit(self):
        """
        TODO
        """

        self.users_preferences_instance = self.users_preferences_instance.fit(X=self.users_pref_dist_df)
        self.users_candidate_instance = self.users_candidate_instance.fit(X=self.users_cand_items_dist_df)
        self.users_recommendation_instance = self.users_recommendation_instance.fit(X=self.users_rec_lists_dist_df)
        if self.conformity_str == Label.KMEANS:
            self.users_preferences_instance = self.users_preferences_instance.predict(X=self.users_pref_dist_df)
            self.users_candidate_instance = self.users_candidate_instance.predict(X=self.users_cand_items_dist_df)
            self.users_recommendation_instance = self.users_recommendation_instance.predict(X=self.users_rec_lists_dist_df)
        elif self.conformity_str == Label.AGGLOMERATIVE:
            self.users_preferences_instance = self.users_preferences_instance.fit_predict(X=self.users_pref_dist_df)
            self.users_candidate_instance = self.users_candidate_instance.fit_predict(X=self.users_cand_items_dist_df)
            self.users_recommendation_instance = self.users_recommendation_instance.fit_predict(X=self.users_rec_lists_dist_df)

    def __silhouette_avg(self):
        user_pref_silhouette_avg = silhouette_score(self.users_pref_dist_df, self.users_preferences_instance)
        users_cand_items_silhouette_avg = silhouette_score(self.users_cand_items_dist_df, self.users_candidate_instance)
        users_rec_lists_silhouette_avg = silhouette_score(self.users_rec_lists_dist_df, self.users_recommendation_instance)

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
            self.users_preferences_instance, self.users_candidate_instance, average='macro'
        )
        user_rec_lists_score_float = jaccard_score(
            self.users_preferences_instance, self.users_recommendation_instance, average='macro'
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
