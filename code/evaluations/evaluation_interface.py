import logging

import pandas as pd

from datasets.registred_datasets import RegisteredDataset
from scikit_pierre.classes.genre import genre_probability_approach
from scikit_pierre.distributions.accessible import distributions_funcs_pandas
from scikit_pierre.measures.accessible import calibration_measures_funcs
from scikit_pierre.metrics.mace import mace
from scikit_pierre.metrics.map import mean_average_precision_map
from scikit_pierre.metrics.mrmc import mrmc
from scikit_pierre.metrics.mrr import mean_reciprocal_rank_map
from settings.labels import Label
from settings.path_dir_file import PathDirFile
from settings.save_and_load import SaveAndLoad

logger = logging.getLogger(__name__)


def applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric='MRR',
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

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

    SaveAndLoad.save_recommender_metric(
        data=results, metric='MRR',
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric='MAP',
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

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

    SaveAndLoad.save_recommender_metric(
        data=results, metric="MAP",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric="MACE",
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

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
    users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
        dataset=dataset_instance.system_name, trial=trial, fold=fold,
        distribution=distribution
    )

    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)
    mace_value = mace(
        users_pref_dist_df, users_recommendation_lists, items_classes_set, dist_func
    )

    results = pd.DataFrame([[
        mace_value
    ]], columns=['MACE'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric="MACE",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"


def applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector):
    """
    Function that apply the evaluation metrics.
    """
    try:
        metric_file = SaveAndLoad.load_recommender_metric(
            metric="MRMC",
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        )
        # Check integrity
        if len(metric_file) == 1:
            return "AlreadyDone"
    except Exception as e:
        logger.error(" - ".join([str(e)]))

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
    users_pref_dist_df = SaveAndLoad.load_user_preference_distribution(
        dataset=dataset_instance.system_name, trial=trial, fold=fold,
        distribution=distribution
    )

    users_recommendation_lists[Label.USER_ID] = users_recommendation_lists[Label.USER_ID].astype(str)
    mrmc_value = mrmc(
        users_pref_dist_df, users_recommendation_lists, items_classes_set, dist_func, fairness_func
    )

    results = pd.DataFrame([[
        mrmc_value
    ]], columns=['MRMC'])

    SaveAndLoad.save_recommender_metric(
        data=results, metric="MRMC",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
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

    SaveAndLoad.save_recommender_metric(
        data=results, metric="TIME",
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        weight=weight, tradeoff=tradeoff, selector=selector
    )
    return "Finished"
