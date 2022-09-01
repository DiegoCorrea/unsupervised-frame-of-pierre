import datetime
import itertools
import logging
import os
import time
from collections import Counter

import pandas as pd
from joblib import Parallel, delayed

from postprocessing.post_processing_step import PostProcessingStep
from settings.constants import Constants
from settings.path_dir_file import PathDirFile
from utils.read_from_terminal import read_input_to_postprocessing_parallel

logger = logging.getLogger(__name__)


def starting_postprocessing(recommender, fold, trial, dataset, tradeoff, distribution, calibration, relevance, weight,
                            selector, list_size, alpha, d):
    """
    Function to starting the processing and run the recommender algorithm.
    """
    path = PathDirFile.get_recommendation_list_file(
        dataset=dataset, recommender=recommender, trial=trial, fold=fold,
        tradeoff=tradeoff, distribution=distribution, fairness=calibration,
        relevance=relevance, tradeoff_weight=weight, select_item=selector)

    # Check integrity.
    if os.path.exists(path):
        try:
            users_recommendation_lists = pd.read_csv(path)
            if dataset == "ml-1m" and len(users_recommendation_lists) == 60400:
                return "AlreadyDone"
            elif dataset == "yahoo-movies" and len(users_recommendation_lists) == 29160:
                return "AlreadyDone"
            else:
                raise "Wrong Lenght"
        except Exception as e:
            logger.error(" - ".join([str(e), path]))

    # Starting the counter
    start_time = time.time()

    # Instancing the post-processing
    pierre = PostProcessingStep(
        recommender=recommender, dataset_name=dataset, trial=trial, fold=fold,
        tradeoff_component=tradeoff, distribution_component=distribution,
        fairness_component=calibration, relevance_component=relevance,
        tradeoff_weight_component=weight, selector_component=selector,
        list_size=list_size, alpha=alpha, d=d
    )
    pierre.run()

    # Finishing the counter
    finish_time = time.time()

    total_time = datetime.timedelta(seconds=finish_time - start_time)
    time_df = pd.DataFrame({"stated_at": [start_time], "finished_at": [finish_time], "total": [total_time]})

    # Saving execution time
    time_df.to_csv(
        PathDirFile.set_postprocessing_time_file(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=calibration,
            relevance=relevance, tradeoff_weight=weight, select_item=selector
        )
    )
    return "Finished"


def main():
    experimental_setup = read_input_to_postprocessing_parallel()
    combination = [experimental_setup['recommenders'], experimental_setup['datasets'],
                   experimental_setup['folds'], experimental_setup['trials'],
                   experimental_setup['tradeoffs'], experimental_setup['relevance_measures'],
                   experimental_setup['distributions'], experimental_setup['selectors'],
                   experimental_setup['weights'], experimental_setup['fairness_measures'],
                   experimental_setup['list_size'], experimental_setup['alpha'], experimental_setup['d']]

    load = Parallel(n_jobs=Constants.N_CORES)(
        delayed(starting_postprocessing)(
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            tradeoff=tradeoff, relevance=relevance, distribution=distribution,
            selector=selector, weight=weight, calibration=calibration,
            list_size=list_size, alpha=alpha, d=d) for
        recommender, dataset, fold, trial, tradeoff, relevance, distribution, selector, weight, calibration, list_size, alpha, d
        in list(itertools.product(*combination)))

    jobs = dict(Counter(load))
    logger.info(jobs)


if __name__ == '__main__':
    """
    Starting the post-processing step
    """
    main()
