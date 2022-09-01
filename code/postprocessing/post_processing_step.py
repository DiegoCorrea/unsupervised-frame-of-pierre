import logging

import pandas as pd

from datasets.candidate_items import CandidateItems
from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.tradeoff.calibration import LinearCalibration, LogarithmBias
from settings.path_dir_file import PathDirFile

logger = logging.getLogger(__name__)


class PostProcessingStep:
    """
    Class to lead with post-processing step
    """

    def __init__(self, recommender, dataset_name, fold, trial, tradeoff_component, distribution_component,
                 fairness_component, relevance_component, tradeoff_weight_component, selector_component,
                 list_size, alpha, d):
        self.recommender = recommender
        self.fold = fold
        self.trial = trial
        self.tradeoff_component = tradeoff_component
        self.distribution_component = distribution_component
        self.fairness_component = fairness_component
        self.relevance_component = relevance_component
        self.tradeoff_weight_component = tradeoff_weight_component
        self.selector_component = selector_component
        self.list_size = list_size
        self.alpha = alpha
        self.d = d
        # Load dataset
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        # Load candidate items set
        self.candidate_items = CandidateItems(recommender=recommender, dataset=dataset_name, trial=trial, fold=fold)
        # Choice the tradeoff
        if self.tradeoff_component == 'LIN':
            self.tradeoff_instance = LinearCalibration(
                users_preferences=self.dataset.get_train_transactions(fold=fold, trial=trial),
                candidate_items=self.candidate_items.get_candidate_items(),
                item_set=self.dataset.get_items()
            )
        elif self.tradeoff_component == 'LOG':
            self.tradeoff_instance = LogarithmBias(
                users_preferences=self.dataset.get_train_transactions(fold=fold, trial=trial),
                candidate_items=self.candidate_items.get_candidate_items(),
                item_set=self.dataset.get_items()
            )

        # Configuring the experimentation
        self.tradeoff_instance.config(
            distribution_component=distribution_component,
            fairness_component=fairness_component,
            relevance_component=relevance_component,
            tradeoff_weight_component=tradeoff_weight_component,
            select_item_component=selector_component,
            list_size=list_size, d=d, alpha=alpha
        )

    def run(self):
        # Execute the instance and get the recommendation list to all users.
        logger.info(">> Running... " + "-".join(
            [self.dataset.system_name, self.recommender,
             self.tradeoff_component, self.distribution_component, self.relevance_component, self.selector_component,
             self.fairness_component, self.tradeoff_weight_component,
             'trial-' + str(self.trial), 'fold-' + str(self.fold)]))
        recommendation_lists = self.tradeoff_instance.fit()
        merged_results_df = pd.concat(recommendation_lists)
        # Save all recommendation lists
        path = PathDirFile.set_recommendation_list_file(
            recommender=self.recommender, dataset=self.dataset.system_name,
            trial=self.trial, fold=self.fold,
            tradeoff=self.tradeoff_component,
            distribution=self.distribution_component,
            fairness=self.fairness_component,
            relevance=self.relevance_component,
            tradeoff_weight=self.tradeoff_weight_component,
            select_item=self.selector_component
        )
        merged_results_df.to_csv(path)
