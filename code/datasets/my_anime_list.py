import os

import numpy as np
import pandas as pd

from datasets.utils.base import Dataset
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class MyAnimeList(Dataset):
    """
    My Anime List dataset.
    This class organize the work with the dataset.
    """
    # Class information.
    dir_name = "mal"
    verbose_name = "My Anime List"
    system_name = "mal"

    # Raw paths.
    dataset_raw_path = "/".join([PathDirFile.RAW_DATASETS_DIR, dir_name])
    raw_transaction_file = "UserAnimeList.csv"
    raw_items_file = "AnimeList.csv"

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
            usecols=['username', 'anime_id', 'my_score'], sep=','
        )
        self.raw_transactions.rename(
            columns={"username": Label.USER_ID, "anime_id": Label.ITEM_ID, "my_score": Label.TRANSACTION_VALUE}, inplace=True
        )

    def clean_transactions(self):
        """
        Cleaning the raw transactions and save as clean transactions.
        """
        super().clean_transactions()

        print("Get Raw Transactions")
        # Load the raw transactions.
        raw_transactions = self.get_raw_transactions()

        # Filter transactions based on the items id list.
        print("Filtering Items")
        filtered_raw_transactions = raw_transactions[
            raw_transactions[Label.ITEM_ID].isin(self.items[Label.ITEM_ID].tolist())]
        del raw_transactions

        # Cut users and set the new data into the instance.
        print("Cut Users")
        self.set_transactions(
            new_transactions=MyAnimeList.cut_users(filtered_raw_transactions))
        del filtered_raw_transactions

        print("Transforming Scores")
        self.transactions[Label.TRANSACTION_VALUE] = np.where(self.transactions[Label.TRANSACTION_VALUE] >= 8, 1, 0)

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
            os.path.join(self.dataset_raw_path, self.raw_items_file),
            sep=',', usecols=['anime_id', 'title', 'genre']
        )
        self.raw_items.rename(
            columns={"anime_id": Label.ITEM_ID, "title": Label.TITLE, "genre": Label.GENRES}, inplace=True
        )

    def clean_items(self):
        """
        Cleaning the raw items and save as clean items.
        """
        # Load the raw items.
        print("Loading Raw Items")
        raw_items_df = self.get_raw_items()

        # Clean the items without information and with the label indicating no genre in the item.
        print("Dropping No Genres")
        genre_clean_items = raw_items_df[raw_items_df[Label.GENRES] != '']
        del raw_items_df
        genre_clean_items[Label.GENRES] = genre_clean_items[Label.GENRES].str.replace(", ", "|")

        # Set the new data into the instance.
        print("Drop Duplicates")
        self.set_items(new_items=genre_clean_items)
        del genre_clean_items
        self.items.drop_duplicates(subset=[Label.ITEM_ID], inplace=True)

        # Save the clean transactions as CSV.
        self.items.to_csv(os.path.join(self.dataset_clean_path, PathDirFile.ITEMS_FILE), index=False)
