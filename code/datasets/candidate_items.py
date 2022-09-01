import pandas as pd

from settings.path_dir_file import PathDirFile


class CandidateItems:
    def __init__(self, recommender, dataset, trial, fold):
        path = PathDirFile.get_candidate_items_file(dataset, recommender, trial, fold)
        self.candidate_items = pd.read_csv(path)

    def get_candidate_items(self):
        return self.candidate_items
