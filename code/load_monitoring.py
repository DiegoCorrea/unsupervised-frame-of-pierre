import itertools
import logging
import os
from collections import Counter

import pandas as pd
from joblib import Parallel, delayed

from settings.constants import Constants
from settings.path_dir_file import PathDirFile
from utils.read_from_terminal import read_input_to_load_monitoring

logger = logging.getLogger(__name__)


def metrics_monitoring(metric, recommender, fold, trial, dataset, tradeoff, distribution, fairness, relevance, weight,
                       selector):
    """
    Function to monitor the metrics.
    """
    exists_file = PathDirFile.get_metric_fold_file_by_name(
        recommender=recommender, dataset=dataset, trial=trial, fold=fold,
        distribution=distribution, fairness=fairness, relevance=relevance,
        tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
        filename=metric + '.csv'
    )
    try:
        # Check integrity
        if os.path.exists(exists_file):
            metric_file = pd.read_csv(exists_file)
            if len(metric_file) == 1:
                return "Done"
            else:
                return "Warning"
    except Exception as e:
        # logger.error(" - ".join([str(e), exists_file]))
        return "Error"
    return "Future"


def postprocessing_monitoring(recommender, fold, trial, dataset, tradeoff, distribution, calibration, relevance, weight,
                              selector):
    """
    Function to monitor the postprocessing.
    """
    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=calibration,
        relevance=relevance, tradeoff_weight=weight, select_item=selector)
    if os.path.exists(path):
        try:
            users_recommendation_lists = pd.read_csv(path)
            if dataset == "ml-1m" and len(users_recommendation_lists) == 60400:
                return "Done"
            elif dataset == "yahoo-movies" and len(users_recommendation_lists) == 29160:
                return "Done"
            else:
                return "Warning"
        except Exception:
            return "Error"
    return "Future"


def main():
    experimental_setup = read_input_to_load_monitoring()
    combination = [experimental_setup['recommenders'], experimental_setup['datasets'],
                   experimental_setup['folds'], experimental_setup['trials'],
                   experimental_setup['tradeoffs'], experimental_setup['relevance_measures'],
                   experimental_setup['distributions'], experimental_setup['selectors'],
                   experimental_setup['weights'], experimental_setup['fairness_measures']]

    all_combinations = list(itertools.product(*combination))
    if experimental_setup['step'] == "POSTPROCESSING":
        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(postprocessing_monitoring)(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                tradeoff=tradeoff, relevance=relevance, distribution=distribution,
                selector=selector, weight=weight, calibration=calibration) for
            recommender, dataset, fold, trial, tradeoff, relevance, distribution, selector, weight, calibration
            in all_combinations)
    else:
        load = Parallel(n_jobs=Constants.N_CORES)(
            delayed(metrics_monitoring)(
                metric=experimental_setup['metric'],
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                tradeoff=tradeoff, relevance=relevance, distribution=distribution,
                selector=selector, weight=weight, fairness=calibration) for
            recommender, dataset, fold, trial, tradeoff, relevance, distribution, selector, weight, calibration
            in list(itertools.product(*combination)))

    jobs = dict(Counter(load))

    if 'Done' in jobs:
        done = jobs['Done']
        print("Done: ", round((done / len(all_combinations)) * 100, 1), "% - ", done, ' of ', len(all_combinations))

    if 'Future' in jobs:
        process = jobs['Future']
        print("Future: ", round((process / len(all_combinations)) * 100, 1), "% - ", process, ' of ',
              len(all_combinations))

    if 'Warning' in jobs:
        warning = jobs['Warning']
        print("Warning: ", round((warning / len(all_combinations)) * 100, 1), "% - ", warning, ' of ',
              len(all_combinations))

    if 'Error' in jobs:
        error = jobs['Error']
        print("Error: ", round((error / len(all_combinations)) * 100, 1), "% - ", error, ' of ', len(all_combinations))


if __name__ == '__main__':
    """
    Starting the load monitoring
    
    python load_monitoring.py -step=POSTPROCESSING --dataset=ml-1m --recommender=SVD
    python load_monitoring.py -step=METRICS -metric=MAP --dataset=ml-1m --recommender=SVD
    """
    main()
