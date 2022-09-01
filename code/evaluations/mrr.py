from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.metrics.mrr import mean_reciprocal_rank
from settings.constants import Constants
from settings.path_dir_file import PathDirFile
import pandas as pd


def mrr_interface(recommender, dataset,
                  distribution, fairness, relevance, weight, tradeoff, selector):
    # Load dataset
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    mrr_values = []
    for trial in list(range(1, Constants.N_TRIAL_VALUE + 1)):
        for fold in list(range(1, Constants.K_FOLDS_VALUE + 1)):
            path = PathDirFile.set_recommendation_list_file(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=weight, select_item=selector)
            users_recommendation_list = pd.read_csv(path)
            users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
            mrr_values.append(mean_reciprocal_rank(users_recommendation_list, users_test_items))
    return sum(mrr_values)/len(mrr_values)
