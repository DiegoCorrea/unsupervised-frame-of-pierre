import os

import pandas as pd

from datasets.utils.base import Dataset
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class MovielensOneMillion(Dataset):
    """
    Movielens 1Million dataset.
    This class organize the work with the dataset.
    """
    # Class information.
    dir_name = "ml-1m"
    verbose_name = "Movielens One Million"
    system_name = "ml-1m"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "ratings.dat"
    raw_items_file = "movies.dat"

    # Clean paths.
    dataset_clean_path = "/".join([PathDirFile.CLEAN_DATASETS_DIR, dir_name])

    # ######################################### #
    # ############## Constructor ############## #
    # ######################################### #

    def __init__(self):
        """
        Class constructor. Firstly call the super constructor and after start personalized things.
        """
        super().__init__()

    # ######################################### #
    # ############# Transactions ############## #
    # ######################################### #

    def load_raw_transactions(self):
        """
        Load raw transactions into the instance variable.
        """
        self.raw_transactions = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_transaction_file),
            names=[Label.USER_ID, Label.ITEM_ID, Label.TRANSACTION_VALUE, Label.TIME],
            engine='python', sep='::'
        )

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions()

        # Filter transactions based on the items id list.
        filtered_raw_transactions = raw_transactions[
            raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]

        # Cut users and set the new data into the instance.
        self.set_transactions(
            new_transactions=MovielensOneMillion.cut_users(filtered_raw_transactions))

        # Save the clean transactions as CSV.
        self.transactions.to_csv(
            os.path.join(self.dataset_clean_path, PathDirFile.TRANSACTIONS_FILE),
            index=False
        )

    # ######################################### #
    # ################# Items ################# #
    # ######################################### #

    def load_raw_items(self):
        """
        Load Raw Items into the instance variable.
        """
        self.raw_items = pd.read_csv(
            os.path.join(self.dataset_raw_path, self.raw_items_file), engine='python',
            sep='::', names=[Label.ITEM_ID, Label.TITLE, Label.GENRES], encoding='ISO-8859-1'
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        raw_items_df.dropna(inplace=True)
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != '(no genres listed)']

        # Set the new data into the instance.
        self.set_items(new_items=genre_clean_items)
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.dataset_clean_path, PathDirFile.ITEMS_FILE), index=False)
