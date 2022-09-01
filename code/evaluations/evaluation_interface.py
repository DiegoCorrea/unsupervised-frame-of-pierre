import logging
import os

import pandas as pd
from numpy import mean

from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.classes.genre import genre_probability_approach
from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.metrics.mace import mace
from scikit_pierre.metrics.map import mean_average_precision_map
from scikit_pierre.metrics.mrmc import mrmc
from scikit_pierre.metrics.mrr import mean_reciprocal_rank_map
from settings.constants import Constants
from settings.path_dir_file import PathDirFile

logger = logging.getLogger(__name__)


def evaluation_interface(recommender, dataset, distribution, fairness, relevance, weight, tradeoff, selector):
    # Load dataset
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)
    fairness_func = calibration_measures_funcs(measure=fairness)

    mrr_values = []
    map_values = []
    mrmc_values = []
    mace_values = []

    for trial in list(range(1, Constants.N_TRIAL_VALUE + 1)):
        for fold in list(range(1, Constants.K_FOLDS_VALUE + 1)):
            print("Trial: ", trial, " | ", "Fold: ", fold, " <> ",
                  "-".join([recommender, tradeoff, distribution, fairness, relevance, selector, weight]))
            path = PathDirFile.get_recommendation_list_file(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, tradeoff_weight=weight, select_item=selector)
            users_recommendation_lists = pd.read_csv(path)

            # Rank Metrics
            users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
            mrr_values.append(mean_reciprocal_rank_map(users_recommendation_lists, users_test_items))
            map_values.append(mean_average_precision_map(users_recommendation_lists, users_test_items))

            # Calibration Metrics
            users_preference_set = dataset_instance.get_train_transactions(trial=trial, fold=fold)
            users_target_dist = pd.concat(list(map(
                lambda pref: dist_func(user_pref_set=pref[1], item_classes_set=items_classes_set),
                users_preference_set.groupby(by=["USER_ID"])
            )))
            mrmc_values.append(mrmc(
                users_target_dist, users_recommendation_lists, items_classes_set, dist_func, fairness_func
            ))
            mace_values.append(mace(
                users_target_dist, users_recommendation_lists, items_classes_set, dist_func
            ))
    return {'MRR': mean(mrr_values), 'MAP': mean(map_values), 'MRMC': mean(mrmc_values), 'MACE': mean(mace_values)}


# ############################################################################################## #
# ############################################################################################## #
def applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_metric_fold_file_by_name(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='MRR.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    mrr_value = mean_reciprocal_rank_map(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        mrr_value
    ]], columns=['MRR'])

    results.to_csv(
        PathDirFile.set_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename='MRR.csv'
        ), index=False
    )
    return "Finished"


def applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_metric_fold_file_by_name(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='MAP.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    users_test_items = dataset_instance.get_test_transactions(trial=trial, fold=fold)
    map_value = mean_average_precision_map(users_recommendation_lists, users_test_items)

    results = pd.DataFrame([[
        map_value
    ]], columns=['MAP'])

    results.to_csv(
        PathDirFile.set_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename='MAP.csv'
        ), index=False
    )
    return "Finished"


def applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_metric_fold_file_by_name(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='MACE.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    dataset_instance = RegisteredDataset.load_dataset(dataset)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Executing
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)

    users_preference_set = dataset_instance.get_train_transactions(trial=trial, fold=fold)
    users_target_dist = pd.concat(list(map(
        lambda pref: dist_func(user_pref_set=pref[1], item_classes_set=items_classes_set),
        users_preference_set.groupby(by=["USER_ID"])
    )))
    mace_value = mace(
        users_target_dist, users_recommendation_lists, items_classes_set, dist_func
    )

    results = pd.DataFrame([[
        mace_value
    ]], columns=['MACE'])

    results.to_csv(
        PathDirFile.set_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename='MACE.csv'
        ), index=False
    )
    return "Finished"


def applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    exists_file = PathDirFile.get_metric_fold_file_by_name(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename='MRMC.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e), exists_file]))

    # Load dataset
    dataset_instance = RegisteredDataset.load_dataset(dataset)
    items_set = dataset_instance.get_items()
    items_classes_set = genre_probability_approach(item_set=items_set)

    dist_func = distributions_funcs_pandas(distribution)
    fairness_func = calibration_measures_funcs(measure=fairness)

    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector
    )
    users_recommendation_lists = pd.read_csv(path)

    # Calibration Metrics
    users_preference_set = dataset_instance.get_train_transactions(trial=trial, fold=fold)
    users_target_dist = pd.concat(list(map(
        lambda pref: dist_func(user_pref_set=pref[1], item_classes_set=items_classes_set),
        users_preference_set.groupby(by=["USER_ID"])
    )))
    mrmc_value = mrmc(
        users_target_dist, users_recommendation_lists, items_classes_set, dist_func, fairness_func
    )

    results = pd.DataFrame([[
        mrmc_value
    ]], columns=['MRMC'])

    results.to_csv(
        PathDirFile.set_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename='MRMC.csv'
        ), index=False
    )
    return "Finished"


def execution_time_fold(recommender, dataset, trial, fold,
                        distribution, fairness, relevance, weight, tradeoff, selector):
    path = PathDirFile.get_postprocessing_time_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=fairness,
        relevance=relevance, tradeoff_weight=weight, select_item=selector)
    execution_time_df = pd.read_csv(path)
    time_values = execution_time_df['finished_at'] - execution_time_df['stated_at']

    results = pd.DataFrame([[
        time_values
    ]], columns=['TIME'])

    results.to_csv(
        PathDirFile.set_metric_fold_file_by_name(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
            filename='TIME.csv'
        ), index=False
    )
    return "Finished"
