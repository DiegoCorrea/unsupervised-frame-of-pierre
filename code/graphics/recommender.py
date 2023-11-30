import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from settings.charts import ChartsConfig
from settings.labels import Label
from settings.path_dir_file import PathDirFile


class SingleRecommenderGraphics:
    """
    TODO
    """

    @staticmethod
    def weight_by_metric_histogram(data: DataFrame, dataset_name: str, metric: str):
        """
        TODO
        """

        X_axis = np.arange(len(Label.CONST_WEIGHT))

        plt.bar(X_axis, data[Label.EVALUATION_METRICS].tolist(), 0.4)

        plt.xticks(X_axis, Label.CONST_WEIGHT)
        plt.xlabel("Trafeoff Weight")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend()

        # Pasta para salvar a figura
        filename = "_".join([metric, "histogram", "CONST", ".png"])
        file_dir = PathDirFile.set_graphics_file(dataset_name, filename)
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')

    @staticmethod
    def weight_by_metric_all_datasets(data: dict, y_label: str, metric: str):
        """
        TODO
        """
        plt.grid(True)
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        plt.xlabel("Tradeoff Weight", fontsize=18)
        plt.ylabel(y_label, fontsize=24)

        n = len(Label.CONST_WEIGHT)
        datasets = ['ml-1m', 'yahoo-movies', 'twitter_movies']
        datasets_name = ["Movielens", "Yahoo Movies", "Twitter Movies"]
        for dataset, name, m, l in zip(datasets, datasets_name, ChartsConfig.markers_list[:n], ChartsConfig.line_style_list[:n]):
            df = data[dataset]
            plt.plot(
                [str(x) for x in df[Label.TRADEOFF_WEIGHT_LABEL].tolist()],
                df[Label.EVALUATION_METRICS].abs().tolist(), alpha=0.5, linestyle=l, marker=m,
                label=str(name), linewidth=4
            )

        lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.20), ncol=3, prop={'size': 18})
        plt.xticks(rotation=30)

        # Pasta para salvar a figura
        filename = "_".join([metric, "all_datasets", ".png"])
        file_dir = PathDirFile.set_graphics_file("all", filename)
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')
