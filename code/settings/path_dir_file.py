import os
from pathlib import Path


class PathDirFile:
    # Base Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.as_posix()

    # Basic Paths
    DATA_DIR = BASE_DIR + "/data"
    LOG_DIR = BASE_DIR + '/logs/'
    RESULTS_DIR = BASE_DIR + "/results"

    # Data Paths
    DATASETS_DIR = BASE_DIR + "/data/datasets"
    RAW_DATASETS_DIR = BASE_DIR + "/data/datasets/raw"
    CLEAN_DATASETS_DIR = BASE_DIR + "/data/datasets/clean"
    TIME_DIR = BASE_DIR + '/data/time'

    # Search Param Path
    SEARCH_PARAMS_DIR = BASE_DIR + "/code/settings/recommender_param"

    # Results Path
    RESULTS_METRICS_DIR = RESULTS_DIR + "/metrics"
    RESULTS_DECISION_DIR = RESULTS_DIR + "/decision"
    RESULTS_GRAPHICS_DIR = RESULTS_DIR + "/graphics"
    RESULTS_DATASET_GRAPHICS_DIR = RESULTS_GRAPHICS_DIR + "/dataset"

    # File
    TRAIN_FILE = 'train.csv'
    TEST_FILE = 'test.csv'
    TRANSACTIONS_FILE = 'transactions.csv'
    ITEMS_FILE = 'items.csv'
    RECOMMENDER_LIST_FILE = "recommendation_list.csv"
    CANDIDATE_ITEMS_FILE = "candidate_items.csv"
    TIME_FILE = "TIME.csv"
    METRICS_FILE = "metrics.csv"
    SYSTEM_METRICS_FILE = "system_metrics.csv"
    DECISION_FILE = 'decision.csv'

    # ########################################################################################### #
    # Pre-processing step methods
    @staticmethod
    def set_preprocessing_time_file(dataset: str) -> str:
        """
        Method to set the file path, which deal with the preprocessing execution time.

        :param dataset: A string that's representing the dataset name.

        :return: A string like data/app/{dataset}/preprocessing/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'preprocessing'])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def get_preprocessing_time_file(dataset: str) -> str:
        """
        Method to get the file path, which deal with the preprocessing execution time.

        :param dataset: A string that's representing the dataset name.

        :return: A string like data/app/{dataset}/preprocessing/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'preprocessing'])
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def set_log_preprocessing_path(dataset: str) -> str:
        """
        Log directory. This method is to deal with the preprocessing step log.

        :param dataset: A string that's representing the dataset name.

        :return: A string like logs/preprocessing/{dataset}/.
        """
        save_in_dir = "/".join([PathDirFile.LOG_DIR, 'preprocessing', dataset])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/'

    # ########################################################################################### #
    # Search step methods
    @staticmethod
    def set_hyperparameter_file(dataset: str, recommender: str) -> str:
        """
        Method to set the file path, which deal with the hyperparameter values founded in the Search Step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.

        :return: A string like code/settings/recommender_param/{dataset}/{recommender}.json.
        """
        save_in_dir = "/".join([PathDirFile.SEARCH_PARAMS_DIR, dataset])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, recommender + ".json"])

    @staticmethod
    def get_hyperparameter_file(dataset: str, recommender: str) -> str:
        """
        Method to get the file path, which deal with the hyperparameter values founded in the Search Step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.

        :return: A string like code/settings/recommender_param/{dataset}/{recommender}.json.
        """
        save_in_dir = "/".join([PathDirFile.SEARCH_PARAMS_DIR, dataset])
        return "/".join([save_in_dir, recommender + ".json"])

    @staticmethod
    def set_search_time_file(dataset: str, recommender: str) -> str:
        """
        Method to set the file path, which deal with the search execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.

        :return: A string like data/app/{dataset}/time/search/{recommender}/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'search', recommender])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def get_search_time_file(dataset: str, recommender: str) -> str:
        """
        Method to get the file path, which deal with the search execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.

        :return: A string like data/app/{dataset}/time/search/{recommender}/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'search', recommender])
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def set_log_search_path(dataset: str, recommender: str) -> str:
        """
        Log directory. This method is to deal with the log in the search step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.

        :return: A string like logs/search/{dataset}/.
        """
        save_in_dir = "/".join([PathDirFile.LOG_DIR, 'search', dataset, recommender])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/'

    # ########################################################################################### #
    # Processing step methods
    @staticmethod
    def set_processing_time_file(dataset: str, recommender: str, trial: int, fold: int) -> str:
        """
        Method to set the file path, which deal with the processing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.

        :return: A string like data/app/{dataset}/time/processing/{recommender}/trial-{trial}/fold-{fold}/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'processing', recommender,
                               'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def get_processing_time_file(dataset: str, recommender: str, trial: int, fold: int) -> str:
        """
        Method to get the file path, which deal with the processing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.

        :return: A string like data/app/{dataset}/time/processing/{recommender}/trial-{trial}/fold-{fold}/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'processing', recommender,
                               'trial-' + str(trial), 'fold-' + str(fold)])
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def set_candidate_items_file(dataset: str, recommender: str, trial: int, fold: int) -> str:
        """
        Method to set the candidate items path, which deal with the candidate items set from the recommender algorithm.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.

        :return: A string like data/app/{dataset}/candidate_items/{recommender}/trial-{trial}/fold-{fold}/candidate_items.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'candidate_items', recommender,
                               'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.CANDIDATE_ITEMS_FILE])

    @staticmethod
    def get_candidate_items_file(dataset: str, recommender: str, trial: int, fold: int) -> str:
        """
        Method to set the candidate items path, which deal with the candidate items set from the recommender algorithm.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.

        :return: A string like data/app/{dataset}/candidate_items/{recommender}/trial-{trial}/fold-{fold}/candidate_items.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'candidate_items', recommender,
                               'trial-' + str(trial), 'fold-' + str(fold)])
        return "/".join([save_in_dir, PathDirFile.CANDIDATE_ITEMS_FILE])

    @staticmethod
    def set_log_processing_path(dataset: str, recommender: str, trial: int, fold: int) -> str:
        """
        Log directory. This method is to deal with the log in the processing step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.

        :return: A string like logs/processing/{dataset}/{recommender}/trial-{trial}/fold-{fold}/.
        """
        save_in_dir = "/".join([PathDirFile.LOG_DIR, 'processing', dataset, recommender,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/'

    # ########################################################################################### #
    # Post-processing step methods
    @staticmethod
    def set_recommendation_list_file(dataset: str, recommender: str, trial: int, fold: int,
                                     tradeoff: str, distribution: str, fairness: str, relevance: str,
                                     tradeoff_weight: str, select_item: str) -> str:
        """
        Method to set the file path, which deal with the recommendation lists from the post-processing step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like data/app/{dataset}/recommendation_lists/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/trial-{trial}/fold-{fold}/.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'recommendation_lists', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.RECOMMENDER_LIST_FILE])

    @staticmethod
    def get_recommendation_list_file(dataset: str, recommender: str, trial: int, fold: int,
                                     tradeoff: str, distribution: str, fairness: str, relevance: str,
                                     tradeoff_weight: str, select_item: str) -> str:
        """
        Method to get the file path, which deal with the recommendation lists from the post-processing step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like data/app/{dataset}/recommendation_lists/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/trial-{trial}/fold-{fold}/.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'recommendation_lists', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        return "/".join([save_in_dir, PathDirFile.RECOMMENDER_LIST_FILE])

    @staticmethod
    def set_postprocessing_time_file(dataset: str, recommender: str, trial: int, fold: int,
                                     tradeoff: str, distribution: str, fairness: str, relevance: str,
                                     tradeoff_weight: str, select_item: str) -> str:
        """
        Method to set the file path, which deal with the postprocessing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like data/app/{dataset}/time/processing/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'postprocessing', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def get_postprocessing_time_file(dataset: str, recommender: str, trial: int, fold: int,
                                     tradeoff: str, distribution: str, fairness: str, relevance: str,
                                     tradeoff_weight: str, select_item: str) -> str:
        """
        Method to get the file path, which deal with the postprocessing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like data/app/{dataset}/time/processing/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/ime.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'time', 'postprocessing', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        return "/".join([save_in_dir, PathDirFile.TIME_FILE])

    @staticmethod
    def set_log_postprocessing_path(dataset: str, recommender: str, trial: int, fold: int,
                                    tradeoff: str, distribution: str, fairness: str, relevance: str,
                                    tradeoff_weight: str, select_item: str) -> str:
        """
        Log directory. This method is to deal with the log in the postprocessing step.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like logs/postprocessing/{dataset}/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/.
        """
        save_in_dir = "/".join([PathDirFile.LOG_DIR, 'postprocessing', dataset, recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/'

    # ########################################################################################### #
    # Final Results
    @staticmethod
    def set_metric_file(recommender: str, dataset: str, distribution: str, fairness: str, relevance: str,
                        tradeoff_weight: str, tradeoff: str, select_item: str) -> str:
        """
        Method to set the file path, which deal with the metric results.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like results/metrics/postprocessing/{dataset}/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/metrics.csv
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_METRICS_DIR, dataset, recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/' + PathDirFile.SYSTEM_METRICS_FILE

    @staticmethod
    def get_metric_file(recommender: str, dataset: str, distribution: str, fairness: str, relevance: str,
                        tradeoff_weight: str, tradeoff: str, select_item: str) -> str:
        """
        Method to get the file path, which deal with the metric results.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like results/metrics/postprocessing/{dataset}/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/metrics.csv
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_METRICS_DIR, dataset, recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight])
        return save_in_dir + '/' + PathDirFile.SYSTEM_METRICS_FILE

    @staticmethod
    def set_metric_fold_file(dataset: str, recommender: str, trial: int, fold: int,
                             tradeoff: str, distribution: str, fairness: str, relevance: str,
                             tradeoff_weight: str, select_item: str) -> str:
        """
        Method to set the file path, which deal with the postprocessing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like data/app/{dataset}/time/processing/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/time.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'metrics', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, PathDirFile.METRICS_FILE])

    @staticmethod
    def get_metric_fold_file(dataset: str, recommender: str, trial: int, fold: int,
                             tradeoff: str, distribution: str, fairness: str, relevance: str,
                             tradeoff_weight: str, select_item: str) -> str:
        """
        Method to get the file path, which deal with the postprocessing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.

        :return: A string like data/app/{dataset}/metrics/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/metrics.csv.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'metrics', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        return "/".join([save_in_dir, PathDirFile.METRICS_FILE])

    @staticmethod
    def set_metric_fold_file_by_name(dataset: str, recommender: str, trial: int, fold: int,
                                     tradeoff: str, distribution: str, fairness: str, relevance: str,
                                     tradeoff_weight: str, select_item: str, filename: str) -> str:
        """
        Method to set the file path, which deal with the postprocessing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.
        :param filename:

        :return: A string like data/app/{dataset}/time/processing/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/{filename}.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'metrics', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return "/".join([save_in_dir, filename])

    @staticmethod
    def get_metric_fold_file_by_name(dataset: str, recommender: str, trial: int, fold: int,
                                     tradeoff: str, distribution: str, fairness: str, relevance: str,
                                     tradeoff_weight: str, select_item: str, filename: str) -> str:
        """
        Method to get the file path, which deal with the postprocessing step execution time.

        :param dataset: A string that's representing the dataset name.
        :param recommender: A string that's representing the recommender algorithm name.
        :param trial: The trial number.
        :param fold: The fold number.
        :param tradeoff: The tradeoff balance component name.
        :param distribution: The distribution component name.
        :param fairness: The fairness measure name.
        :param relevance: The relevance measure name.
        :param tradeoff_weight: The tradeoff weight component name.
        :param select_item: The select item algorithm name.
        :param filename:

        :return: A string like data/app/{dataset}/metrics/{recommender}/{tradeoff_component}/
        {distribution_component}/{relevance_component}/{selector_component}/{fairness_component}/{tradeoff_weight_component}/
        trial-{trial}/fold-{fold}/{filename}.
        """
        save_in_dir = "/".join([PathDirFile.DATA_DIR, 'app', dataset, 'metrics', recommender,
                                tradeoff, distribution, relevance, select_item, fairness, tradeoff_weight,
                                'trial-' + str(trial), 'fold-' + str(fold)])
        return "/".join([save_in_dir, filename])

    # ########################################################################################### #
    # Decision
    @staticmethod
    def set_decision_file(dataset: str) -> str:
        """
        Method to set the file path, which deal with the decision protocol.

        :param dataset: A string that's representing the dataset name.

        :return: A string like results/decision/{dataset}/decision.csv
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_DECISION_DIR, dataset])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/' + PathDirFile.DECISION_FILE

    @staticmethod
    def get_decision_file(dataset: str) -> str:
        """
        Method to get the file path, which deal with the decision protocol.

        :param dataset: A string that's representing the dataset name.

        :return: A string like results/decision/{dataset}/decision.csv
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_DECISION_DIR, dataset])
        return save_in_dir + '/' + PathDirFile.DECISION_FILE

    # ########################################################################################### #
    # Graphics
    @staticmethod
    def set_graphics_file(dataset: str, filename: str) -> str:
        """
        Method to set the file path, which deal with the graphics files.

        :param dataset: A string that's representing the dataset name.
        :param filename: A string that's representing the graphic file name.

        :return: A string like results/graphics/results/{dataset}/{filename}
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_GRAPHICS_DIR, "metrics", dataset])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/' + filename

    @staticmethod
    def get_graphics_file(dataset: str, filename: str) -> str:
        """
        Method to get the file path, which deal with the graphics files.

        :param dataset: A string that's representing the dataset name.
        :param filename: A string that's representing the graphic file name.

        :return: A string like results/graphics/results/{dataset}/{filename}
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_GRAPHICS_DIR, "results", dataset])
        return save_in_dir + '/' + filename

    @staticmethod
    def set_dataset_graphics_file(dataset: str, filename: str) -> str:
        """
        Method to set the file path, which deal with the graphics files.

        :param dataset: A string that's representing the dataset name.
        :param filename: A string that's representing the graphic file name.

        :return: A string like results/decision/{dataset}/decision.csv
        """
        save_in_dir = "/".join([PathDirFile.RESULTS_DATASET_GRAPHICS_DIR, dataset])
        if not os.path.exists(save_in_dir):
            os.makedirs(save_in_dir)
        return save_in_dir + '/' + filename
