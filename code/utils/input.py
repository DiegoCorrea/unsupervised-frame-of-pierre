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
        experimental_setup['cluster'] = Label.DEFAULT_CLUSTERING
        experimental_setup['distribution'] = Label.DEFAULT_DISTRIBUTION
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
            print("All params possibilities are: --dataset, --recommender, --trial and --fold.")
            print("Example: python step3_processing.py --dataset=yahoo-movies --recommender=SVD --trial=1 --fold=1")
            exit(1)
        return experimental_setup
