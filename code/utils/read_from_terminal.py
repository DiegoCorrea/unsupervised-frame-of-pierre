import sys

from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from settings.labels import Label


def read_input_to_preprocessing() -> dict:
    """
    Function to read the settings from the terminal. The possible options are:

    - dataset can be: ml-1m, yahoo-movies (see the registered datasets).

    - fold can be: 2, 3 or higher.

    - trial can be: 1, 2, 3 or higher.

    :return: A dict with the input settings.
    """
    experimental_setup = dict()
    # Experimental setup information
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
                experimental_setup['dataset'] = value
            # read number of folds
            elif param == '--n_folds':
                if int(value) <= 2:
                    print('The low accepted value is 3!')
                    exit(1)
                experimental_setup['n_folds'] = int(value)
            # read number of trials
            elif param == '--n_trials':
                if int(value) < 1:
                    print('Just positive number is accepted!')
                    exit(1)
                experimental_setup['n_trials'] = int(value)
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
        print("All params possibilities are: --dataset, --n_folds and --n_trials.")
        print("Example: python preprocessing_entry.py --dataset=yahoo-movies")
        exit(1)
    return experimental_setup


def read_input_to_searching() -> dict:
    """
    Function to read the settings from the terminal. The possible options are:

    - recommender can be: SVD, SVD++, NMF and others.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    :return: A dict with the input settings.
    """
    input_options = dict()
    input_options['recommender'] = Label.DEFAULT_REC
    input_options['dataset'] = RegisteredDataset.DEFAULT_DATASET
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            if param == '--recommender':
                if value not in Label.REGISTERED_RECOMMENDERS:
                    print('Recommender not found!')
                    exit(1)
                input_options['recommender'] = value
            # read the dataset to be used
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered!')
                    exit(1)
                input_options['dataset'] = value
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
        print("All params possibilities are: --recommender and --dataset.")
        print("Example: python searches_entry.py --dataset=yahoo-movies --recommender=SVD")
        exit(1)
    return input_options


def read_input_to_processing() -> dict:
    """
    Function to read the settings from the terminal. The possible options are:

    - recommender can be: SVD, SVD++, NMF and others.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

    - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

    :return: A dict with the input settings.
    """
    input_options = dict()
    input_options['recommender'] = Label.DEFAULT_REC
    input_options['dataset'] = RegisteredDataset.DEFAULT_DATASET
    input_options['fold'] = 1
    input_options['trial'] = 1
    if len(sys.argv) > 2:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            if param == '--recommender':
                if value not in Label.REGISTERED_RECOMMENDERS:
                    print('Recommender not found!')
                    exit(1)
                input_options['recommender'] = value
            # read the dataset to be used
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered!')
                    exit(1)
                input_options['dataset'] = value
            # read the fold number
            elif param == '--fold':
                if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                    print('Fold out of range!')
                    exit(1)
                input_options['fold'] = value
            # read the trial number
            elif param == '--trial':
                if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                    print('Fold out of range!')
                    exit(1)
                input_options['trial'] = value
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
        print("All params possibilities are: --dataset, --recommender, --trial and --fold.")
        print("Example: python processing_entry.py --dataset=yahoo-movies --recommender=SVD")
        exit(1)
    return input_options


def read_input_to_postprocessing():
    """
    Function to read the settings from the terminal. The possible options are:

    - recommender can be: SVD, SVD++, NMF and others.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

    - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

    - node is the machine name.

    - tradeoff

    - calibration

    - relevance

    - weight

    - distribution

    - selector

    - list_size

    - alpha

    - d

    :return: A dict with the input settings.
    """
    input_options = dict()
    # Recommender settings
    input_options['recommender'] = Label.DEFAULT_REC
    input_options['dataset'] = RegisteredDataset.DEFAULT_DATASET
    input_options['fold'] = 1
    input_options['trial'] = 1
    # post-processing settings
    input_options['tradeoff'] = Label.DEFAULT_TRADEOFF
    input_options['calibration'] = Label.DEFAULT_CALIBRATION
    input_options['relevance'] = Label.DEFAULT_RELEVANCE
    input_options['weight'] = Label.DEFAULT_WEIGHT
    input_options['distribution'] = Label.DEFAULT_DISTRIBUTION
    input_options['selector'] = Label.DEFAULT_SELECTOR
    input_options['list_size'] = Constants.RECOMMENDATION_LIST_SIZE
    # Extra settings to post-processing
    input_options['extra'] = dict(alpha=Constants.ALPHA_VALUE, d=Constants.DIMENSION_VALUE)

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            # read the recommender
            if param == '--recommender':
                if value not in Label.REGISTERED_RECOMMENDERS:
                    print('Recommender not found! Options is:')
                    print(Label.REGISTERED_RECOMMENDERS)
                    exit(1)
                input_options['recommender'] = value
            # read dataset
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered! Options is:')
                    print(RegisteredDataset.DATASET_LIST)
                    exit(1)
                input_options['dataset'] = value
            # read the fold number
            elif param == '--fold':
                if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                    print('Fold out of range! Options is:')
                    print(list(range(1, Constants.K_FOLDS_VALUE + 1)))
                    exit(1)
                input_options['fold'] = value
            # read the trial number
            elif param == '--trial':
                if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                    print('Trial out of range! Options is:')
                    print(list(range(1, Constants.N_TRIAL_VALUE + 1)))
                    exit(1)
                input_options['trial'] = value
            elif param == '--tradeoff':
                if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
                    print('Tradeoff not registered! Options is:')
                    print(Label.ACCESSIBLE_TRADEOFF_LIST)
                    exit(1)
                input_options['tradeoff'] = value
            elif param == '--relevance':
                if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
                    print('Relevance not registered! Options is:')
                    print(Label.ACCESSIBLE_RELEVANCE_LIST)
                    exit(1)
                input_options['relevance'] = value
            elif param == '--calibration':
                if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
                    print('Calibration measure not registered! Options is:')
                    print(Label.ACCESSIBLE_CALIBRATION_LIST)
                    exit(1)
                input_options['calibration'] = value
            elif param == '--distribution':
                if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                    print('Distribution not registered! Options is:')
                    print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
                    exit(1)
                input_options['distribution'] = value
            elif param == '--selector':
                if value not in Label.ACCESSIBLE_SELECTOR_LIST:
                    print('Selector not registered! Options is:')
                    print(Label.ACCESSIBLE_SELECTOR_LIST)
                    exit(1)
                input_options['selector'] = value
            elif param == '--weight':
                if value not in Label.ACCESSIBLE_WEIGHT_LIST:
                    print('Tradeoff Weight not registered! Options is:')
                    print(Label.ACCESSIBLE_WEIGHT_LIST)
                    exit(1)
                input_options['weight'] = value
            elif param == '--list_size':
                if int(value) <= 0 or int(value) > Constants.RECOMMENDATION_LIST_SIZE:
                    print('List size out of range! Options is:')
                    print(list(range(1, Constants.RECOMMENDATION_LIST_SIZE + 1)))
                    exit(1)
                input_options['list_size'] = value
            elif param == '--alpha':
                input_options['extra']['alpha'] = value
            elif param == '--d':
                input_options['extra']['d'] = value
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
    return input_options


def read_input_to_postprocessing_parallel():
    """
    Function to read the settings from the terminal. The possible options are:

    - recommender can be: SVD, SVD++, NMF and others.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

    - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

    - node is the machine name.

    - tradeoff

    - calibration

    - relevance

    - weight

    - distribution

    - selector

    - list_size

    - alpha

    - d

    :return: A dict with the input settings.
    """
    input_options = dict()
    # Recommender settings
    input_options['recommenders'] = Label.REGISTERED_RECOMMENDERS
    input_options['datasets'] = RegisteredDataset.DATASET_LIST
    input_options['folds'] = list(range(1, Constants.K_FOLDS_VALUE + 1))
    input_options['trials'] = list(range(1, Constants.N_TRIAL_VALUE + 1))
    # post-processing settings
    input_options['tradeoffs'] = Label.ACCESSIBLE_TRADEOFF_LIST
    input_options['fairness_measures'] = Label.ACCESSIBLE_CALIBRATION_LIST
    input_options['relevance_measures'] = Label.ACCESSIBLE_RELEVANCE_LIST
    input_options['weights'] = Label.ACCESSIBLE_WEIGHT_LIST
    input_options['distributions'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
    input_options['selectors'] = Label.ACCESSIBLE_SELECTOR_LIST
    input_options['list_size'] = [Constants.RECOMMENDATION_LIST_SIZE]
    # Extra settings to post-processing
    input_options['alpha'] = [Constants.ALPHA_VALUE]
    input_options['d'] = [Constants.DIMENSION_VALUE]

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            # read the recommender
            if param == '--recommender':
                if value not in Label.REGISTERED_RECOMMENDERS:
                    print('Recommender not found! Options is:')
                    print(Label.REGISTERED_RECOMMENDERS)
                    exit(1)
                input_options['recommenders'] = [value]
            # read dataset
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered! Options is:')
                    print(RegisteredDataset.DATASET_LIST)
                    exit(1)
                input_options['datasets'] = [value]
            # read the fold number
            elif param == '--fold':
                if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                    print('Fold out of range! Options is:')
                    print(list(range(1, Constants.K_FOLDS_VALUE + 1)))
                    exit(1)
                input_options['folds'] = [value]
            # read the trial number
            elif param == '--trial':
                if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                    print('Trial out of range! Options is:')
                    print(list(range(1, Constants.N_TRIAL_VALUE + 1)))
                    exit(1)
                input_options['trials'] = [value]
            elif param == '--tradeoff':
                if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
                    print('Tradeoff not registered! Options is:')
                    print(Label.ACCESSIBLE_TRADEOFF_LIST)
                    exit(1)
                input_options['tradeoffs'] = [value]
            elif param == '--relevance':
                if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
                    print('Relevance not registered! Options is:')
                    print(Label.ACCESSIBLE_RELEVANCE_LIST)
                    exit(1)
                input_options['relevance_measures'] = [value]
            elif param == '--calibration':
                if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
                    print('Calibration measure not registered! Options is:')
                    print(Label.ACCESSIBLE_CALIBRATION_LIST)
                    exit(1)
                input_options['fairness_measures'] = [value]
            elif param == '--distribution':
                if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                    print('Distribution not registered! Options is:')
                    print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
                    exit(1)
                input_options['distributions'] = [value]
            elif param == '--selector':
                if value not in Label.ACCESSIBLE_SELECTOR_LIST:
                    print('Selector not registered! Options is:')
                    print(Label.ACCESSIBLE_SELECTOR_LIST)
                    exit(1)
                input_options['selectors'] = [value]
            elif param == '--weight':
                if value not in Label.ACCESSIBLE_WEIGHT_LIST:
                    print('Tradeoff Weight not registered! Options is:')
                    print(Label.ACCESSIBLE_WEIGHT_LIST)
                    exit(1)
                input_options['weights'] = [value]
            elif param == '--list_size':
                if int(value) <= 0 or int(value) > Constants.RECOMMENDATION_LIST_SIZE:
                    print('List size out of range! Options is:')
                    print(list(range(1, Constants.RECOMMENDATION_LIST_SIZE + 1)))
                    exit(1)
                input_options['list_size'] = [value]
            elif param == '--alpha':
                input_options['alpha'] = [value]
            elif param == '--d':
                input_options['d'] = [value]
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
    return input_options


def read_input_to_load_monitoring():
    """
    Function to read the settings from the terminal. The possible options are:

    - recommender can be: SVD, SVD++, NMF and others.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

    - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

    - node is the machine name.

    - tradeoff

    - calibration

    - relevance

    - weight

    - distribution

    - selector

    - list_size

    - alpha

    - d

    :return: A dict with the input settings.
    """
    input_options = dict()
    input_options['metric'] = ['MAP']
    input_options['step'] = "POSTPROCESSING"
    # Recommender settings
    input_options['recommenders'] = ['SVD']
    input_options['datasets'] = RegisteredDataset.DATASET_LIST
    input_options['folds'] = list(range(1, Constants.K_FOLDS_VALUE + 1))
    input_options['trials'] = list(range(1, Constants.N_TRIAL_VALUE + 1))
    # post-processing settings
    input_options['tradeoffs'] = Label.ACCESSIBLE_TRADEOFF_LIST
    input_options['fairness_measures'] = Label.ACCESSIBLE_CALIBRATION_LIST
    input_options['relevance_measures'] = Label.ACCESSIBLE_RELEVANCE_LIST
    input_options['weights'] = Label.ACCESSIBLE_WEIGHT_LIST
    input_options['distributions'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
    input_options['selectors'] = Label.ACCESSIBLE_SELECTOR_LIST

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            # read the recommender
            if param == '--recommender':
                if value not in Label.REGISTERED_RECOMMENDERS:
                    print('Recommender not found! Options is:')
                    print(Label.REGISTERED_RECOMMENDERS)
                    exit(1)
                input_options['recommenders'] = [value]
            # read dataset
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered! Options is:')
                    print(RegisteredDataset.DATASET_LIST)
                    exit(1)
                input_options['datasets'] = [value]
            # read the fold number
            elif param == '--fold':
                if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                    print('Fold out of range! Options is:')
                    print(list(range(1, Constants.K_FOLDS_VALUE + 1)))
                    exit(1)
                input_options['folds'] = [value]
            # read the trial number
            elif param == '--trial':
                if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                    print('Trial out of range! Options is:')
                    print(list(range(1, Constants.N_TRIAL_VALUE + 1)))
                    exit(1)
                input_options['trials'] = [value]
            elif param == '--tradeoff':
                if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
                    print('Tradeoff not registered! Options is:')
                    print(Label.ACCESSIBLE_TRADEOFF_LIST)
                    exit(1)
                input_options['tradeoffs'] = [value]
            elif param == '--relevance':
                if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
                    print('Relevance not registered! Options is:')
                    print(Label.ACCESSIBLE_RELEVANCE_LIST)
                    exit(1)
                input_options['relevance_measures'] = [value]
            elif param == '--calibration':
                if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
                    print('Calibration measure not registered! Options is:')
                    print(Label.ACCESSIBLE_CALIBRATION_LIST)
                    exit(1)
                input_options['fairness_measures'] = [value]
            elif param == '--distribution':
                if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                    print('Distribution not registered! Options is:')
                    print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
                    exit(1)
                input_options['distributions'] = [value]
            elif param == '--selector':
                if value not in Label.ACCESSIBLE_SELECTOR_LIST:
                    print('Selector not registered! Options is:')
                    print(Label.ACCESSIBLE_SELECTOR_LIST)
                    exit(1)
                input_options['selectors'] = [value]
            elif param == '--weight':
                if value not in Label.ACCESSIBLE_WEIGHT_LIST:
                    print('Tradeoff Weight not registered! Options is:')
                    print(Label.ACCESSIBLE_WEIGHT_LIST)
                    exit(1)
                input_options['weights'] = [value]
            elif param == '-step':
                if value not in ["PROCESSING", "POSTPROCESSING", "METRICS"]:
                    print('System step not registered! Options is:')
                    print(["PROCESSING", "POSTPROCESSING", "METRICS"])
                    exit(1)
                input_options['step'] = value
            elif param == '-metric':
                if value not in ['MAP', 'MRR', 'MACE', 'MRMC', 'TIME']:
                    print('Metric not found! Options is:')
                    print(['MAP', 'MRR', 'MACE', 'MRMC', 'TIME'])
                    exit(1)
                input_options['metric'] = value
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
    return input_options


def read_input_chart_analises() -> dict:
    """
    Function to read the settings from the terminal. The possible options are:

    - opt can be: CHART, ANALYZE.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    :return: A dict with the input settings.
    """
    input_options = dict()
    input_options['opt'] = "CHART"
    input_options['dataset'] = RegisteredDataset.DATASET_LIST
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            if param == '-opt':
                if value not in ['CHART', "ANALYZE"]:
                    print('Option not found!')
                    exit(1)
                input_options['opt'] = value
            # read the dataset to be used
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered!')
                    exit(1)
                input_options['dataset'] = [value]
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
        print("All params possibilities are: -opt and --dataset.")
        print("Example: python step7_charts_analises.py -opt=CHART --dataset=yahoo-movies")
        exit(1)
    return input_options


def read_input_to_metrics():
    """
    Function to read the settings from the terminal. The possible options are:

    - metric: The possibles values can be MAP, MRR, MACE, MRMC, TIME, e.g., "--metrics=MAP"

    - recommender can be: SVD, SVD++, NMF and others.

    - dataset can be: ml-1m, yahoo-movies and others (see the registered datasets).

    - fold can be: 1, 2, 3 and others (based on the preprocessing n_folds).

    - trial can be: 1, 2, 3 and others (based on the preprocessing n_trials).

    - node is the machine name.

    - tradeoff

    - calibration

    - relevance

    - weight

    - distribution

    - selector

    :return: A dict with the input settings.
    """
    input_options = dict()
    input_options['metrics'] = ['ALL']
    # Recommender settings
    input_options['recommenders'] = Label.REGISTERED_RECOMMENDERS
    input_options['datasets'] = RegisteredDataset.DATASET_LIST
    input_options['folds'] = list(range(1, Constants.K_FOLDS_VALUE + 1))
    input_options['trials'] = list(range(1, Constants.N_TRIAL_VALUE + 1))
    # post-processing settings
    input_options['tradeoffs'] = Label.ACCESSIBLE_TRADEOFF_LIST
    input_options['fairness_measures'] = Label.ACCESSIBLE_CALIBRATION_LIST
    input_options['relevance_measures'] = Label.ACCESSIBLE_RELEVANCE_LIST
    input_options['weights'] = Label.ACCESSIBLE_WEIGHT_LIST
    input_options['distributions'] = Label.ACCESSIBLE_DISTRIBUTION_LIST
    input_options['selectors'] = Label.ACCESSIBLE_SELECTOR_LIST

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            param, value = arg.split('=')
            # read the metric
            if param == '-metric':
                if value not in ['MAP', 'MRR', 'MACE', 'MRMC', 'TIME', 'RANK', 'CALIBRATION', 'ALL']:
                    print('Metric not found! Options is:')
                    print(['MAP', 'MRR', 'MACE', 'MRMC', 'TIME', 'RANK', 'CALIBRATION', 'ALL'])
                    exit(1)
                input_options['metrics'] = [value]
            # read the recommender
            elif param == '--recommender':
                if value not in Label.REGISTERED_RECOMMENDERS:
                    print('Recommender not found! Options is:')
                    print(Label.REGISTERED_RECOMMENDERS)
                    exit(1)
                input_options['recommenders'] = [value]
            # read dataset
            elif param == '--dataset':
                if value not in RegisteredDataset.DATASET_LIST:
                    print('Dataset not registered! Options is:')
                    print(RegisteredDataset.DATASET_LIST)
                    exit(1)
                input_options['datasets'] = [value]
            # read the fold number
            elif param == '--fold':
                if int(value) <= 0 or int(value) > Constants.K_FOLDS_VALUE:
                    print('Fold out of range! Options is:')
                    print(list(range(1, Constants.K_FOLDS_VALUE + 1)))
                    exit(1)
                input_options['folds'] = [value]
            # read the trial number
            elif param == '--trial':
                if int(value) <= 0 or int(value) > Constants.N_TRIAL_VALUE:
                    print('Trial out of range! Options is:')
                    print(list(range(1, Constants.N_TRIAL_VALUE + 1)))
                    exit(1)
                input_options['trials'] = [value]
            elif param == '--tradeoff':
                if value not in Label.ACCESSIBLE_TRADEOFF_LIST:
                    print('Tradeoff not registered! Options is:')
                    print(Label.ACCESSIBLE_TRADEOFF_LIST)
                    exit(1)
                input_options['tradeoffs'] = [value]
            elif param == '--relevance':
                if value not in Label.ACCESSIBLE_RELEVANCE_LIST:
                    print('Relevance not registered! Options is:')
                    print(Label.ACCESSIBLE_RELEVANCE_LIST)
                    exit(1)
                input_options['relevance_measures'] = [value]
            elif param == '--calibration':
                if value not in Label.ACCESSIBLE_CALIBRATION_LIST:
                    print('Calibration measure not registered! Options is:')
                    print(Label.ACCESSIBLE_CALIBRATION_LIST)
                    exit(1)
                input_options['fairness_measures'] = [value]
            elif param == '--distribution':
                if value not in Label.ACCESSIBLE_DISTRIBUTION_LIST:
                    print('Distribution not registered! Options is:')
                    print(Label.ACCESSIBLE_DISTRIBUTION_LIST)
                    exit(1)
                input_options['distributions'] = [value]
            elif param == '--selector':
                if value not in Label.ACCESSIBLE_SELECTOR_LIST:
                    print('Selector not registered! Options is:')
                    print(Label.ACCESSIBLE_SELECTOR_LIST)
                    exit(1)
                input_options['selectors'] = [value]
            elif param == '--weight':
                if value not in Label.ACCESSIBLE_WEIGHT_LIST:
                    print('Tradeoff Weight not registered! Options is:')
                    print(Label.ACCESSIBLE_WEIGHT_LIST)
                    exit(1)
                input_options['weights'] = [value]
            else:
                print("The parameter {} is not configured in this feature.".format(param))
    else:
        print("More information are needed!")
    return input_options
