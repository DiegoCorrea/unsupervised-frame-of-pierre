import pandas as pd

from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.classes.genre import genre_probability_approach
from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.metrics.mrmc import mrmc
from settings.constants import Constants
from settings.path_dir_file import PathDirFile


def mrmc_interface(recommender, dataset,
                   distribution, fairness, relevance, weight, tradeoff, selector):
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)
    fairness_func = calibration_measures_funcs(measure=fairness)

    values = []
    for trial in list(range(1, Constants.N_TRIAL_VALUE + 1)):
        for fold in list(range(1, Constants.K_FOLDS_VALUE + 1)):
            print("Trial: ", trial, " | ", "Fold: ", fold)
            recommendation_list_path = PathDirFile.get_recommendation_list_file(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=weight, select_item=selector
            )
            users_recommendation_lists = pd.read_csv(recommendation_list_path)

            users_preference_set = dataset_instance.get_train_transactions(trial=trial, fold=fold)

            users_target_dist = pd.concat([
                dist_func(
                    user_pref_set=user_preference_set,
                    item_classes_set=items_classes_set
                ) for user_id, user_preference_set in users_preference_set.groupby(by=["USER_ID"])
            ])
            print("Tamanho: ", len(users_target_dist))
            mrmc_value = mrmc(
                users_target_dist, users_recommendation_lists, items_classes_set, dist_func, fairness_func
            )
            values.append(mrmc_value)
            print(values)
    return sum(values) / len(values)
