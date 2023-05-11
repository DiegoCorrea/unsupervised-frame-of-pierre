import pandas as pd

from datasets.registred_datasets import RegisteredDataset
from pierre.classes.genre import genre_probability_approach
from pierre.distributions.accessible import distributions_funcs_pandas
from pierre.metrics.mace import mace
from settings.constants import Constants
from settings.path_dir_file import PathDirFile


def mace_interface(recommender, dataset,
                   distribution, fairness, relevance, weight, tradeoff, selector):
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)

    values = []
    for trial in list(range(1, Constants.N_TRIAL_VALUE + 1)):
        for fold in list(range(1, Constants.K_FOLDS_VALUE + 1)):
            recommendation_list_path = PathDirFile.get_recommendation_list_file(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=weight, select_item=selector
            )
            users_recommendation_lists = pd.read_csv(recommendation_list_path)

            users_preference_set = dataset_instance.get_train_transactions(trial=trial, fold=fold)

            users_target_dist = pd.concat([
                dist_func(
                    user_id=user_id,
                    user_pref_set=users_preference_set[users_preference_set['USER_ID'] == user_id],
                    item_classes_set=items_classes_set
                ) for user_id in users_preference_set['USER_ID'].unique().tolist()
            ])
            users_target_dist.fillna(0.0, inplace=True)

            mace_value = mace(
                users_target_dist=users_target_dist, users_recommendation_lists=users_recommendation_lists,
                items_classes_set=items_classes_set, dist_func=dist_func
            )
            values.append(mace_value)
    return sum(values) / len(values)
