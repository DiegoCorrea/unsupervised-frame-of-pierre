import sys

from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from settings.labels import Label


class Input:
    """
    TODO: Docstring
    """

    @staticmethod
    def step1() -> dict:
        """
        Method to read the settings from the terminal/keyboard. The possible options are:

        - opt: TODO

        - dataset can be: ml-1m, yahoo-movies (see the registered datasets).

        - n_folds can be: 1, 2, 3 or higher.

        - n_trials can be: 1, 2, 3 or higher.

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        # Experimental setup information
        experimental_setup['opt'] = Label.DATASET_SPLIT
        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['n_folds'] = Constants.K_FOLDS_VALUE
        experimental_setup['n_trials'] = Constants.N_TRIAL_VALUE
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')
                # read dataset
                if param == '--dataset':
                    if value not in RegisteredDataset.DATASET_LIST:
                        print('Dataset not registered!')
                        exit(1)
                    experimental_setup['dataset'] = str(value)
                # read number of folds
                elif param == '--n_folds':
                    if int(value) < 3:
                        print('The lower accepted value is 3!')
                        exit(1)
                    experimental_setup['n_folds'] = int(value)
                # read number of trials
                elif param == '--n_trials':
                    if int(value) < 1:
                        print('Only positive numbers are accepted!')
                        exit(1)
                    experimental_setup['n_trials'] = int(value)
                elif param == '-opt':
                    if param not in Label.PREPROCESSING_OPTS:
                        print('This option does not exists!')
                        exit(1)
                    experimental_setup['opt'] = str(value)

                else:
                    print(f"The parameter {param} is not configured in this feature.")
        else:
            print("More information are needed!")
            print("All params possibilities are: -opt --dataset, --n_folds and --n_trials.")
            print("Example: python step1_preprocessing.py -opt=SPLIT --dataset=ml-1m --n_trials=10")
            exit(1)
        return experimental_setup

    @staticmethod
    def step2() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - recommender can be: SVD, SVD++, NMF and others.

        - cluster: TODO: Docstring

        - distribution: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['opt'] = Label.RECOMMENDER
        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['recommender'] = Label.DEFAULT_REC
        experimental_setup['cluster'] = Label.DEFAULT_CLUSTERING
        experimental_setup['distribution'] = Label.DEFAULT_DISTRIBUTION
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')
                if param == '--recommender':
                    if value not in Label.REGISTERED_RECOMMENDERS:
                        print('Recommender not found!')
                        exit(1)
                    experimental_setup['recommender'] = value
                # read the dataset to be used
                elif param == '--dataset':
                    if value not in RegisteredDataset.DATASET_LIST:
                        print('Dataset not registered!')
                        exit(1)
                    experimental_setup['dataset'] = value
                elif param == '--cluster':
                    if value not in Label.REGISTERED_CLUSTERS:
                        print('Cluster algorithm not registered!')
                        exit(1)
                    experimental_setup['cluster'] = value
                elif param == '--distribution':
                    if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                        print('Distribution not found!')
                        exit(1)
                    experimental_setup['distribution'] = value
                elif param == '-opt':
                    if value not in Label.SEARCH_OPTS:
                        print(f'This option does not exists! {value}')
                        exit(1)
                    experimental_setup['opt'] = str(value)
                else:
                    print(f"The parameter {param} is not configured in this feature.")
        else:
            print("More information are needed!")
            print("All params possibilities are: -opt, --dataset, --recommender, --cluster, --distribution.")
            print("Example: python step2_random_search.py -opt-RECOMMENDER --recommender=SVD --dataset=ml-1m")
            exit(1)
        return experimental_setup

    @staticmethod
    def step3() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

        - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

        - cluster: TODO: Docstring

        - distribution: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['opt'] = Label.RECOMMENDER
        experimental_setup['recommender'] = Label.DEFAULT_REC
        experimental_setup['dataset'] = RegisteredDataset.DEFAULT_DATASET
        experimental_setup['fold'] = 1
        experimental_setup['trial'] = 1

        if len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')
                if param == '--recommender':
                    if value not in Label.REGISTERED_RECOMMENDERS:
                        print('Recommender not found! All possibilities are:')
                        print(Label.REGISTERED_RECOMMENDERS)
                        exit(1)
                    experimental_setup['recommender'] = value
                # read the dataset to be used
                elif param == '--dataset':
                    if value not in RegisteredDataset.DATASET_LIST:
                        print('Dataset not registered! All possibilities are:')
                        print(RegisteredDataset.DATASET_LIST)
                        exit(1)
                    experimental_setup['dataset'] = value
                # read the fold number
                elif param == '--fold':
                    if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                        print('Fold out of range!')
                        exit(1)
                    experimental_setup['fold'] = value
                # read the trial number
                elif param == '--trial':
                    if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                        print('Fold out of range!')
                        exit(1)
                    experimental_setup['trial'] = value
                elif param == '-opt':
                    if value not in Label.SEARCH_OPTS:
                        print(f'This option does not exists! {value}')
                        exit(1)
                    experimental_setup['opt'] = str(value)
                else:
                    print(f"The parameter {param} is not configured in this feature.")
        else:
            print("More information are needed!")
            print("All params possibilities are: --dataset, --recommender, --trial and --fold.")
            print("Example: python step3_processing.py --dataset=yahoo-movies --recommender=SVD --trial=1 --fold=1")
            exit(1)
        return experimental_setup

    @staticmethod
    def step5() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

        - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

        - cluster: TODO: Docstring

        - tradeoff: TODO: Docstring

        - calibration: TODO: Docstring

        - relevance: TODO: Docstring

        - weight: TODO: Docstring

        - distribution: TODO: Docstring

        - selector: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['opt'] = Label.EVALUATION_METRICS
        experimental_setup['metrics'] = Label.REGISTERED_METRICS

        experimental_setup['recommender'] = Label.REGISTERED_RECOMMENDERS
        experimental_setup['cluster'] = Label.REGISTERED_CLUSTERS

        experimental_setup['dataset'] = [RegisteredDataset.DEFAULT_DATASET]
        experimental_setup['fold'] = list(range(1, Constants.K_FOLDS_VALUE + 1))
        experimental_setup['trial'] = list(range(1, Constants.N_TRIAL_VALUE + 1))

        experimental_setup['tradeoff'] = Label.ACCESSIBLE_TRADEOFF_LIST
        experimental_setup['fairness'] = Label.ACCESSIBLE_CALIBRATION_LIST
        experimental_setup['relevance'] = Label.ACCESSIBLE_RELEVANCE_LIST
        experimental_setup['weight'] = Label.ACCESSIBLE_WEIGHT_LIST
        experimental_setup['distribution'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
        experimental_setup['selector'] = Label.ACCESSIBLE_SELECTOR_LIST

        if len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')
                if param == '-opt':
                    if value not in Label.METRIC_OPT:
                        print(f'This option does not exists! {value}... All possibilities are:')
                        print(Label.METRIC_OPT)
                        exit(1)
                    experimental_setup['opt'] = str(value)
                elif param == '-metric':
                    if value not in Label.REGISTERED_METRICS:
                        print('Metric not found! Options is:')
                        print(Label.REGISTERED_METRICS)
                        exit(1)
                    experimental_setup['metrics'] = [value]
                elif param == '-cluster':
                    if value not in Label.REGISTERED_CLUSTERS:
                        print('Cluster algorithm not registered! All possibilities are:')
                        print(Label.REGISTERED_CLUSTERS)
                        exit(1)
                    experimental_setup['cluster'] = [value]
                elif param == '--recommender':
                    if value not in Label.REGISTERED_RECOMMENDERS:
                        print('Recommender not found! All possibilities are:')
                        print(Label.REGISTERED_RECOMMENDERS)
                        exit(1)
                    experimental_setup['recommender'] = [value]
                # read the dataset to be used
                elif param == '--dataset':
                    if value not in RegisteredDataset.DATASET_LIST:
                        print('Dataset not registered! All possibilities are:')
                        print(RegisteredDataset.DATASET_LIST)
                        exit(1)
                    experimental_setup['dataset'] = [value]
                # read the fold number
                elif param == '--fold':
                    if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                        print('Fold out of range!')
                        exit(1)
                    experimental_setup['fold'] = [value]
                # read the trial number
                elif param == '--trial':
                    if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                        print('Fold out of range!')
                        exit(1)
                    experimental_setup['trial'] = [value]
                elif param == '--tradeoff':
                    if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
                        print('Tradeoff not registered! Options is:')
                        print(Label.ACCESSIBLE_TRADEOFF_LIST)
                        exit(1)
                    experimental_setup['tradeoff'] = [value]
                elif param == '--relevance':
                    if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
                        print('Relevance not registered! Options is:')
                        print(Label.ACCESSIBLE_RELEVANCE_LIST)
                        exit(1)
                    experimental_setup['relevance'] = [value]
                elif param == '--calibration':
                    if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
                        print('Calibration measure not registered! Options is:')
                        print(Label.ACCESSIBLE_CALIBRATION_LIST)
                        exit(1)
                    experimental_setup['fairness'] = [value]
                elif param == '--distribution':
                    if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                        print('Distribution not registered! Options is:')
                        print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
                        exit(1)
                    experimental_setup['distribution'] = [value]
                elif param == '--selector':
                    if value not in Label.ACCESSIBLE_SELECTOR_LIST:
                        print('Selector not registered! Options is:')
                        print(Label.ACCESSIBLE_SELECTOR_LIST)
                        exit(1)
                    experimental_setup['selector'] = [value]
                elif param == '--weight':
                    if value not in Label.ACCESSIBLE_WEIGHT_LIST:
                        print('Tradeoff Weight not registered! Options is:')
                        print(Label.ACCESSIBLE_WEIGHT_LIST)
                        exit(1)
                    experimental_setup['weight'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        else:
            print("More information are needed!")
            exit(1)
        return experimental_setup

    @staticmethod
    def step6() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt: TODO: Docstring

        - metrics: TODO: Docstring

        - recommender can be: SVD, SVD++, NMF and others.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        - cluster: TODO: Docstring

        - tradeoff: TODO: Docstring

        - calibration: TODO: Docstring

        - relevance: TODO: Docstring

        - weight: TODO: Docstring

        - distribution: TODO: Docstring

        - selector: TODO: Docstring

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['opt'] = Label.EVALUATION_METRICS
        experimental_setup['metrics'] = Label.REGISTERED_METRICS

        experimental_setup['recommender'] = Label.REGISTERED_RECOMMENDERS
        experimental_setup['conformity'] = Label.REGISTERED_CLUSTERS

        experimental_setup['dataset'] = RegisteredDataset.DATASET_LIST

        experimental_setup['tradeoff'] = Label.ACCESSIBLE_TRADEOFF_LIST
        experimental_setup['fairness'] = Label.ACCESSIBLE_CALIBRATION_LIST
        experimental_setup['relevance'] = Label.ACCESSIBLE_RELEVANCE_LIST
        experimental_setup['weight'] = Label.ACCESSIBLE_WEIGHT_LIST
        experimental_setup['distribution'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
        experimental_setup['selector'] = Label.ACCESSIBLE_SELECTOR_LIST

        if len(sys.argv) > 2:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')
                if param == '-opt':
                    if value not in Label.METRIC_OPT:
                        print(f'This option does not exists! {value}... All possibilities are:')
                        print(Label.METRIC_OPT)
                        exit(1)
                    experimental_setup['opt'] = str(value)
                elif param == '-metric':
                    if value not in Label.REGISTERED_METRICS:
                        print('Metric not found! Options is:')
                        print(Label.REGISTERED_METRICS)
                        exit(1)
                    experimental_setup['metrics'] = [value]
                elif param == '-conformity':
                    if value not in Label.REGISTERED_CLUSTERS:
                        print('Cluster algorithm not registered! All possibilities are:')
                        print(Label.REGISTERED_CLUSTERS)
                        exit(1)
                    experimental_setup['conformity'] = [value]
                elif param == '--recommender':
                    if value not in Label.REGISTERED_RECOMMENDERS:
                        print('Recommender not found! All possibilities are:')
                        print(Label.REGISTERED_RECOMMENDERS)
                        exit(1)
                    experimental_setup['recommender'] = [value]
                # read the dataset to be used
                elif param == '--dataset':
                    if value not in RegisteredDataset.DATASET_LIST:
                        print('Dataset not registered! All possibilities are:')
                        print(RegisteredDataset.DATASET_LIST)
                        exit(1)
                    experimental_setup['dataset'] = [value]
                elif param == '--tradeoff':
                    if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
                        print('Tradeoff not registered! Options is:')
                        print(Label.ACCESSIBLE_TRADEOFF_LIST)
                        exit(1)
                    experimental_setup['tradeoff'] = [value]
                elif param == '--relevance':
                    if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
                        print('Relevance not registered! Options is:')
                        print(Label.ACCESSIBLE_RELEVANCE_LIST)
                        exit(1)
                    experimental_setup['relevance'] = [value]
                elif param == '--calibration':
                    if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
                        print('Calibration measure not registered! Options is:')
                        print(Label.ACCESSIBLE_CALIBRATION_LIST)
                        exit(1)
                    experimental_setup['fairness'] = [value]
                elif param == '--distribution':
                    if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                        print('Distribution not registered! Options is:')
                        print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
                        exit(1)
                    experimental_setup['distribution'] = [value]
                elif param == '--selector':
                    if value not in Label.ACCESSIBLE_SELECTOR_LIST:
                        print('Selector not registered! Options is:')
                        print(Label.ACCESSIBLE_SELECTOR_LIST)
                        exit(1)
                    experimental_setup['selector'] = [value]
                elif param == '--weight':
                    if value not in Label.ACCESSIBLE_WEIGHT_LIST:
                        print('Tradeoff Weight not registered! Options is:')
                        print(Label.ACCESSIBLE_WEIGHT_LIST)
                        exit(1)
                    experimental_setup['weight'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        else:
            print("More information are needed!")
            exit(1)
        return experimental_setup

    @staticmethod
    def step7() -> dict:
        """
        Function to read the settings from the terminal. The possible options are:

        - opt can be: CHART, ANALYZE.

        - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

        :return: A dict with the input settings.
        """
        experimental_setup = dict()
        experimental_setup['opt'] = Label.EVALUATION_METRICS
        # experimental_setup['metrics'] = Label.REGISTERED_METRICS

        experimental_setup['conformity'] = Label.REGISTERED_CLUSTERS
        experimental_setup['view'] = Label.EVALUATION_VIEWS

        experimental_setup['dataset'] = RegisteredDataset.DATASET_LIST
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                param, value = arg.split('=')
                if param == '-opt':
                    if value not in Label.METRIC_OPT:
                        print(f'This option does not exists! {value}... All possibilities are:')
                        print(Label.METRIC_OPT)
                        exit(1)
                    experimental_setup['opt'] = str(value)
                elif param == '-metric':
                    if value not in Label.REGISTERED_METRICS:
                        print(f'Metric {value} not found! Options is:')
                        print(Label.REGISTERED_METRICS)
                        exit(1)
                    experimental_setup['metrics'] = [value]
                elif param == '-view':
                    if value not in Label.EVALUATION_VIEWS:
                        print(f'View {value} not found!')
                        exit(1)
                    experimental_setup['view'] = value
                # read the dataset to be used
                elif param == '--dataset':
                    if value not in RegisteredDataset.DATASET_LIST:
                        print('Dataset not registered!')
                        exit(1)
                    experimental_setup['dataset'] = [value]
                else:
                    print("The parameter {} is not configured in this feature.".format(param))
        else:
            print("More information are needed!")
            print("All params possibilities are: -opt and --dataset.")
            print("Example: python step7_charts_analises.py -opt=CHART --dataset=yahoo-movies")
            exit(1)
        return experimental_setup
