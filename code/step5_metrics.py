import itertools
import logging
from collections import Counter

from joblib import Parallel, delayed

from evaluations.evaluation_interface import execution_time_fold, applying_map, \
    applying_mrr, applying_mace, applying_mrmc
from settings.constants import Constants
from utils.read_from_terminal import read_input_to_metrics

logger = logging.getLogger(__name__)


def applying_metrics(metrics, recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                     selector):
    """
    Function to apply the evaluation metrics.
    """
    load_monitoring = []
    for m in metrics:
        if m == 'MAP':
            load_monitoring.append(
                applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                             selector)
            )
        elif m == "MRR":
            load_monitoring.append(
                applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                             selector)
            )
        elif m == "MACE":
            load_monitoring.append(
                applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                              selector)
            )
        elif m == "MRMC":
            load_monitoring.append(
                applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                              selector)
            )
        elif m == "TIME":
            execution_time_fold(recommender, dataset, trial, fold,
                                distribution, fairness, relevance, weight, tradeoff, selector)
        elif m == "RANK":
            load_monitoring.append(
                applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                             selector)
            )
            load_monitoring.append(
                applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                             selector)
            )
        elif m == "CALIBRATION":
            load_monitoring.append(
                applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                              selector)
            )
            load_monitoring.append(
                applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                              selector)
            )
        else:
            execution_time_fold(recommender, dataset, trial, fold,
                                distribution, fairness, relevance, weight, tradeoff, selector)
            load_monitoring.append(
                applying_mrr(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                             selector)
            )
            load_monitoring.append(
                applying_map(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                             selector)
            )
            load_monitoring.append(
                applying_mrmc(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                              selector)
            )
            load_monitoring.append(
                applying_mace(recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff,
                              selector)
            )
            break
    return load_monitoring


def main():
    setup_config = read_input_to_metrics()

    combination = [
        setup_config['recommenders'], setup_config['datasets'],
        setup_config['trials'], setup_config['folds'],
        setup_config['distributions'], setup_config['fairness_measures'], setup_config['relevance_measures'],
        setup_config['weights'], setup_config['tradeoffs'], setup_config['selectors']
    ]

    load = Parallel(n_jobs=Constants.N_CORES)(
        delayed(applying_metrics)(
            metrics=setup_config['metrics'],
            recommender=recommender, dataset=dataset, trial=trial, fold=fold,
            distribution=distribution, fairness=fairness, relevance=relevance,
            weight=weight, tradeoff=tradeoff, selector=selector
        ) for recommender, dataset, trial, fold, distribution, fairness, relevance, weight, tradeoff, selector
        in list(itertools.product(*combination)))

    # jobs = dict(Counter(load))
    print(load)


if __name__ == '__main__':
    """
    Starting the metrics
    """
    main()
