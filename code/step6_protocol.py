import itertools
import logging

import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame

from settings.constants import Constants
from settings.path_dir_file import PathDirFile
from utils.read_from_terminal import read_input_to_postprocessing_parallel

logger = logging.getLogger(__name__)


def merge_metrics(recommender: str, dataset: str, trial: int, fold: int,
                  distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str):
    """
    Merge metrics in a single DataFrame

    :param recommender: ...
    :param dataset: ...
    :param trial: ...
    :param fold: ...
    :param distribution: ...
    :param fairness: ...
    :param relevance: ...
    :param weight: ...
    :param tradeoff: ...
    :param selector: ...

    :return: ...
    """

    errors = []
    merged_metrics = pd.DataFrame([])

    for metric in ['MAP', 'MRR', 'MRMC', 'MACE', 'TIME']:
        path = PathDirFile.get_metric_fold_file_by_name(
            dataset=dataset, recommender=recommender, trial=trial, fold=fold,
            tradeoff=tradeoff, distribution=distribution, fairness=fairness,
            relevance=relevance, tradeoff_weight=weight, select_item=selector,
            filename=metric + '.csv'
        )
        try:
            metric_df = pd.read_csv(path)
            merged_metrics[metric] = metric_df[metric]
        except Exception as e:
            # logger.error(" - ".join([str(e), "File does not exist or without content that pandas can read!"]))
            errors.append("Error")
            merged_metrics[metric] = None
            continue

    try:
        merged_metrics.to_csv(
            PathDirFile.set_metric_fold_file_by_name(
                recommender=recommender, dataset=dataset, trial=trial, fold=fold,
                distribution=distribution, fairness=fairness, relevance=relevance,
                tradeoff_weight=weight, tradeoff=tradeoff, select_item=selector,
                filename='ALL_METRICS.csv'
            ), index=False
        )
    except Exception as e:
        # logger.error(" - ".join([str(e), "Error due the merged metrics save!"]))
        errors.append("Error")

    # print(errors)
    # print(merged_metrics)
    return merged_metrics


def load_results(recommender: str, dataset: str,
                 distribution: str, fairness: str, relevance: str, weight: str, tradeoff: str, selector: str
                 ) -> DataFrame:
    """
    Merge metrics in a single DataFrame

    :param recommender: ...
    :param dataset: ...
    :param distribution: ...
    :param fairness: ...
    :param relevance: ...
    :param weight: ...
    :param tradeoff: ...
    :param selector: ...

    :return: ...
    """

    result_list = []
    for trial in range(1, Constants.N_TRIAL_VALUE + 1):
        for fold in range(1, Constants.K_FOLDS_VALUE + 1):
            result_list.append(merge_metrics(
                dataset=dataset, recommender=recommender, trial=trial, fold=fold,
                tradeoff=tradeoff, distribution=distribution, fairness=fairness,
                relevance=relevance, weight=weight, selector=selector
            ))

    metrics = pd.concat(result_list)
    try:
        metrics_av = [metrics[column].sum()
                      if column == "TIME" else metrics[column].mean()
                      for column in metrics.columns.tolist()]
        result = pd.DataFrame([metrics_av], columns=metrics.columns.tolist())
    except Exception as e:
        print(metrics)
    result['COMBINATION'] = "-".join([recommender, tradeoff, distribution, fairness, relevance, selector, weight])
    return result


def main():
    setup_config = read_input_to_postprocessing_parallel()

    for dataset_name in setup_config['datasets']:
        combination = [
            setup_config['recommenders'], [dataset_name],
            setup_config['distributions'], setup_config['fairness_measures'], setup_config['relevance_measures'],
            setup_config['weights'], setup_config['tradeoffs'], setup_config['selectors']
        ]
        output = Parallel(n_jobs=Constants.N_CORES)(
            delayed(load_results)(
                recommender=recommender, dataset=dataset,
                distribution=distribution, fairness=fairness, relevance=relevance,
                weight=weight, tradeoff=tradeoff, selector=selector
            ) for recommender, dataset, distribution, fairness, relevance, weight, tradeoff, selector
            in list(itertools.product(*combination))
        )
        # print(output)
        results = pd.concat(output)
        results['CMC'] = results['MAP']/results['MRMC']
        results['CCE'] = results['MAP']/results['MACE']
        results['PERFORMANCE'] = results['CCE'] + results['CMC']
        results.sort_values(by=['MAP'], ascending=False, inplace=True)
        # print(results)
        results.to_csv(PathDirFile.set_decision_file(dataset_name), index=False)


if __name__ == '__main__':
    """
    Starting the decision protocol
    """
    main()
