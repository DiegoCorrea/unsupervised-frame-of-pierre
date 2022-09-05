import sys

from datasets.registred_datasets import RegisteredDataset
from settings.constants import Constants
from settings.labels import Label


class Input:
    """
    TODO: Docstring
    """

    @staticmethod
    def step1():
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
