import logging

from pandas import DataFrame
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neural_network import BernoulliRBM

from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.classes.genre import genre_probability_approach

from datasets.registred_datasets import RegisteredDataset
from searches.parameters import ConformityParams
from settings.labels import Label
import pandas as pd

from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


class ConformityAlgorithms:
    """
    TODO
    """

    def __init__(self, cluster: str, distribution: str, dataset: str, fold: int, trial: int,
                 recommender: str, tradeoff: str, fairness: str, relevance: str, weight: str, selector: str):
        self.dataset = RegisteredDataset.load_dataset(dataset)

        items_set = self.dataset.get_items()
        items_classes_set = genre_probability_approach(item_set=items_set)
        users_preference_set = self.dataset.get_train_transactions(trial=trial, fold=fold)
        dist_func = distributions_funcs_pandas(distribution)

        self.distribution_instance = calibration_measures_funcs(measure=distribution)
        self.users_target_dist = pd.concat([
            dist_func(
                user_pref_set=users_preference_set[users_preference_set['USER_ID'] == user_id],
                item_classes_set=items_classes_set
            ) for user_id in users_preference_set['USER_ID'].unique().tolist()
        ])

        recommendation_list_path = PathDirFile.get_recommendation_list_file(
            dataset=dataset, recommender=recommender, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, tradeoff_weight=weight, select_item=selector
        )
        self.users_recommendation_lists = pd.read_csv(recommendation_list_path)
        self.users_recommendation_dist = pd.concat([
            dist_func(
                user_pref_set=self.users_recommendation_lists[self.users_recommendation_lists['USER_ID'] == user_id],
                item_classes_set=items_classes_set
            ) for user_id in self.users_recommendation_lists['USER_ID'].unique().tolist()
        ])

        self.cluster_name = cluster
        self.algorithm_instance = None
        self.algorithm_predict_instance = None
        self.params = None
        if cluster == Label.KMEANS:
            self.algorithm_instance = KMeans(n_clusters=3)
        elif cluster == Label.AGGLOMERATIVE:
            self.algorithm_instance = AgglomerativeClustering(n_clusters=3)
        elif cluster == Label.BERNOULLI_RBM:
            self.algorithm_instance = BernoulliRBM(n_components=3)

    def fit(self):
        """
        TODO
        """
        self.algorithm_instance = self.algorithm_instance.fit(X=self.users_target_dist)
        if self.cluster_name == Label.KMEANS:
            self.algorithm_instance = self.algorithm_instance.predict(X=self.users_recommendation_dist)
        elif self.cluster_name == Label.AGGLOMERATIVE:
            self.algorithm_instance = self.algorithm_instance.fit_predict(X=self.users_recommendation_dist)

    def evaluation(
            self, metrics: list,
            cluster: str, recommender: str, dataset: str, trial: int, fold: int,
            distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
    ):
        silhouette_avg = silhouette_score(self.users_target_dist, self.algorithm_instance)
        data = DataFrame([[silhouette_avg]], columns=[Label.SILHOUETTE_SCORE])
        print("Silhouette avg:", silhouette_avg)
        SaveAndLoad.save_conformity_metric(
            data=data, cluster=cluster, metric=Label.SILHOUETTE_SCORE,
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
