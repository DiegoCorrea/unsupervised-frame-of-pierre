from analyses.genres import user_genres_analysis, genre_probability_distribution
from analyses.popularities import count_item_popularity
from datasets.registred_datasets import RegisteredDataset
from graphics.genres import GenreChats
from graphics.long_tail import long_tail_graphic
from settings.labels import Label


class DatasetChart:

    def __init__(self, dataset_name):
        self.dataset = RegisteredDataset.load_dataset(dataset_name)
        self.item_popularity = None

    def item_long_tail(self):
        self.item_popularity = count_item_popularity(self.dataset.get_transactions())
        long_tail_graphic(self.item_popularity, self.dataset.system_name)

    def genres(self):
        complete_transactions_df = self.dataset.get_transactions().merge(self.dataset.get_items(), on=Label.ITEM_ID)
        #
        analysis_of_users_df = user_genres_analysis(complete_transactions_df)
        analysis_of_users_df.sort_values(by=[Label.USER_MODEL_SIZE_LABEL], ascending=[False], inplace=True)
        GenreChats.user_model_size_by_number_of_genres(analysis_of_users_df, self.dataset.system_name)
        #
        users_genre_distr_df = genre_probability_distribution(complete_transactions_df, label=Label.USER_ID)
        print(users_genre_distr_df)
        items_genre_distr_df = genre_probability_distribution(self.dataset.get_items(), label=Label.ITEM_ID)
        print(items_genre_distr_df)
        GenreChats.compare_genre_distribution_bar(users_genre_distr_df, items_genre_distr_df, self.dataset.dir_name)
