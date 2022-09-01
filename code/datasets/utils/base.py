import itertools
import logging
import os

import pandas as pd

from datasets.utils import split
from settings.constants import Constants
from settings.labels import Label
from settings.path_dir_file import PathDirFile

logger = logging.getLogger(__name__)


class Dataset:
    """
    Dataset class to be inherited.
    To be inheritance by all registered dataset.
    """
    # Class information.
    dir_name = ""
    verbose_name = ""
    system_name = ""

    # Raw paths.
    dataset_raw_path = ""
    raw_transaction_file = ""
    raw_items_file = ""

    # Clean paths.
    dataset_clean_path = PathDirFile.CLEAN_DATASETS_DIR

    # ######################################### #
    # ############## Constructor ############## #
    # ######################################### #

    def __init__(self):
        """
        Base Dataset Class Constructor (super).
        """
        # Creating the raw data variables.
        self.raw_transactions = None
        self.raw_items = None

        # Creating the clean data variables.
        self.transactions = None
        self.items = None

        # Fold
        self.train_transaction = None
        self.test_transaction = None

        # Creating the directory to lead with the clean data.
        self.create_clean_dir()

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    # # RAW
    def load_raw_transactions(self):
        """
        Load Raw Transactions.
        The specific implementation is created by the children.
        """
        pass

    def get_raw_transactions(self) -> pd.DataFrame():
        """
        Get method to return the raw data transactions.
        :return: A pandas Dataframe with the dataset raw transactions.
        """
        # If it is the first requisition, load from the file
        if self.raw_transactions is None:
            self.load_raw_transactions()
        return self.raw_transactions

    # # CLEAN
    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        The specific implementation is created by the children.
        """
        pass

    def load_clean_transactions(self):
        """
        Load clean transactions into the instance.
        """
        # TODO: colocar o caminho dentro da classe gerenciadora de caminhos
        self.transactions = pd.read_csv(os.path.join(self.dataset_clean_path, PathDirFile.TRANSACTIONS_FILE))

    def get_transactions(self) -> pd.DataFrame():
        """
        Get method to return the clean data transactions.
        :return: A pandas Dataframe with the dataset clean transactions.
        """
        # If it is the first requisition, load from the file
        if self.transactions is None:
            self.load_clean_transactions()
        return self.transactions

    def set_transactions(self, new_transactions: pd.DataFrame):
        """
        Set method to attribute the new transactions data to the instance.
        :param new_transactions: A pandas Dataframe with the new clean dataset transactions.
        """
        self.transactions = new_transactions

    # TRAIN AND TEST DATA
    def load_train_transactions(self, trial: int, fold: int):
        """
        Load a train transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        """
        # TODO: colocar o caminho dentro da classe gerenciadora de caminhos
        fold_dir = "/".join([self.dataset_clean_path, "trial-" + str(trial), "fold-" + str(fold)])
        self.train_transaction = pd.read_csv(os.path.join(fold_dir, PathDirFile.TRAIN_FILE))

    def get_train_transactions(self, trial: int, fold: int) -> pd.DataFrame:
        """
        Get the train transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        :return: A pandas Dataframe with the train transactions.
        """
        # If it is the first requisition, load from the file
        if self.train_transaction is None:
            self.load_train_transactions(trial, fold)
        return self.train_transaction

    def load_test_transactions(self, trial: int, fold: int):
        """
        Load a test transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        """
        # TODO: colocar o caminho dentro da classe gerenciadora de caminhos
        fold_dir = "/".join([self.dataset_clean_path, "trial-" + str(trial), "fold-" + str(fold)])
        self.test_transaction = pd.read_csv(os.path.join(fold_dir, PathDirFile.TEST_FILE))

    def get_test_transactions(self, trial: int, fold: int):
        """
        Get the test transaction set.
        :param trial: An int that represents a number of an experimental trial.
        :param fold: An int that represents a number of a fold.
        :return: A pandas Dataframe with the test transactions.
        """
        # If it is the first requisition, load from the file
        if self.test_transaction is None:
            self.load_test_transactions(trial, fold)
        return self.test_transaction

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    # RAW
    def load_raw_items(self):
        """
        Load Raw Items.
        The specific implementation is created by the children.
        """
        pass

    def get_raw_items(self) -> pd.DataFrame():
        """
        Get method to return the raw items data.
        :return: A pandas Dataframe with the dataset raw items.
        """
        # If it is the first requisition, load from the file
        if self.raw_items is None:
            self.load_raw_items()
        return self.raw_items

    # CLEAN
    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        The specific implementation is created by the children.
        """
        pass

    def load_clean_items(self):
        """
        Load clean items into the instance.
        """
        # TODO: colocar o caminho dentro da classe gerenciadora de caminhos
        self.items = pd.read_csv(os.path.join(self.dataset_clean_path, PathDirFile.ITEMS_FILE))

    def load_clean_dataset(self):
        """
        Load the clean transactions and items into the instance.
        """
        self.load_clean_transactions()
        self.load_clean_items()

    def get_items(self) -> pd.DataFrame():
        """
        Get method to return the clean items data.
        :return: A pandas Dataframe with the dataset clean items.
        """
        # If is the first requisition, load from the file
        if self.items is None:
            self.load_clean_items()
        return self.items

    def set_items(self, new_items: pd.DataFrame):
        """
        Set method to attribute the new items data to the instance.
        :param new_items: A pandas Dataframe with the new clean dataset items.
        """
        self.items = new_items

    # ######################################### #
    # ################# Other ################# #
    # ######################################### #

    def create_clean_dir(self):
        """
        Create the dataset directory to receive the clean data (pre-processed data).
        """
        # TODO: colocar o caminho dentro da classe gerenciadora de caminhos
        if not os.path.exists(self.dataset_clean_path):
            os.makedirs(self.dataset_clean_path)

    def drop_time_column(self):
        """
        Drop the time column, if needed.
        """
        self.transactions.drop([Label.TIME], inplace=True, axis=1)

    def clean_data(self):
        """
        Caller to clean the transactions and items
        """
        # Extract the items
        self.clean_items()
        # Extract the transactions
        self.clean_transactions()

    def mining_data_and_create_fold(self, n_trials: int = Constants.N_TRIAL_VALUE,
                                    n_folds: int = Constants.K_FOLDS_VALUE):
        """
        The raw dataset is preprocessed and the clean dataset produce n_trials with n_folds.

        :param n_trials: A int that represents a number of experimental trials to create.
        :param n_folds: A int that represents a number of the k folds.
        """
        # Clean and filter the data
        self.clean_data()
        # Creating Folds
        self.create_folds(n_trials=n_trials, n_folds=n_folds)

    def create_folds(self, n_trials: int = Constants.N_TRIAL_VALUE, n_folds: int = Constants.K_FOLDS_VALUE):
        """
        Create all folds to be used by the system.
        The clean dataset produce n_trials with n_folds.

        :param n_trials: A int that represents a number of experimental trials to create.
        :param n_folds: A int that represents a number of the k folds.
        """
        for trial in range(1, n_trials + 1):
            logger.info("+ Preparing trial: " + str(trial))
            results = split.split_with_joblib(transactions_df=self.transactions, trial=trial, n_folds=n_folds)
            train_df = results[0]
            test_df = results[1]
            for k in range(Constants.K_FOLDS_VALUE):
                logger.info("+ + Preparing fold: " + str(k + 1))
                fold_dir = "/".join([self.dataset_clean_path, "trial-" + str(trial), "fold-" + str(k + 1)])
                if not os.path.exists(fold_dir):
                    os.makedirs(fold_dir)

                train_path = os.path.join(fold_dir, PathDirFile.TRAIN_FILE)
                if 'index' in train_df[k].columns.tolist():
                    train_df[k].drop(columns=['index'], inplace=True)
                train_df[k].to_csv(train_path, index=False)

                test_path = os.path.join(fold_dir, PathDirFile.TEST_FILE)
                if 'index' in test_df[k].columns.tolist():
                    test_df[k].drop(columns=['index'], inplace=True)
                test_df[k].to_csv(test_path, index=False)

    @staticmethod
    def cut_users(transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Cleaning the raw transactions and save as clean transactions.
        The specific implementation is created by the children.
        :return: A pandas Dataframe with the users transactions.
        """
        user_counts = transactions[Label.USER_ID].value_counts()
        return transactions[transactions[Label.USER_ID].isin(
            [k for k, v in user_counts.items() if v > Constants.PROFILE_LEN_CUT_VALUE])].copy()

    ################################################
    def raw_data_basic_info(self):
        self.load_raw_items()
        self.load_raw_transactions()

        def classes(item):
            # item_genre = getattr(row, GENRES_LABEL)
            splitted = item.split('|')
            return [c for c in splitted]
        total_of_users = len(self.raw_transactions[Label.USER_ID].unique())
        total_of_items = len(self.raw_items)
        total_of_transactions = len(self.raw_transactions)
        total_of_classes = len(set(list(itertools.chain.from_iterable(list(map(classes, self.raw_items[Label.GENRES].tolist()))))))
        print("RAW DATASET INFORMATION")
        print("Total of Users: ", total_of_users)
        print("Total of Items: ", total_of_items)
        print("Total of Transactions: ", total_of_transactions)
        print("Total of Classes: ", total_of_classes)

    def clean_data_basic_info(self):
        self.load_clean_dataset()

        def classes(item):
            # item_genre = getattr(row, GENRES_LABEL)
            splitted = item.split('|')
            return [c for c in splitted]
        total_of_users = len(self.transactions[Label.USER_ID].unique())
        total_of_items = len(self.items)
        total_of_transactions = len(self.transactions)
        total_of_classes = len(set(list(itertools.chain.from_iterable(list(map(classes, self.items[Label.GENRES].tolist()))))))
        print("CLEAN DATASET INFORMATION")
        print("Total of Users: ", total_of_users)
        print("Total of Items: ", total_of_items)
        print("Total of Transactions: ", total_of_transactions)
        print("Total of Classes: ", total_of_classes)
