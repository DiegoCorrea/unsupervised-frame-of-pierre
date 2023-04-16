import matplotlib.pyplot as plt
import numpy as np

from settings.labels import Label
from settings.path_dir_file import PathDirFile


class ConformityGraphics:
    @staticmethod
    def silhouette_group_bar(results, dataset_name, conformity_algos, metric):
        df = results.groupby(by=[Label.CONFORMITY_DIST_MEANING])
        # groups_label = [Label.USERS_PREF, Label.USERS_CAND_ITEMS, Label.USERS_REC_LISTS]
        groups_label = []
        means_dict = {label: [] for label in conformity_algos}
        for group in df:
            print(group[0])
            groups_label.append(group[0])
            for index, row in group[1].iterrows():
                _, algo_name, _, _, _, _, _, _ = row['COMBINATION'].split("-")
                means_dict[algo_name].append(round(abs(row[metric]) * 100, 0))
        print(means_dict)

        width = 1/(len(means_dict) + 2)  # the width of the bars
        x = np.arange(len(groups_label))

        # plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()
        i = 1
        sides = 1
        if len(conformity_algos) % 2 == 0:
            for algo in conformity_algos:
                if i % 2 == 0:
                    bar = ax.bar(x - (width * sides), means_dict[algo], width, label=algo)
                    sides += 1
                else:
                    bar = ax.bar(x + (width * sides), means_dict[algo], width, label=algo)
                ax.bar_label(bar, padding=3)
                i += 1
        else:
            for algo in conformity_algos:
                if i % 3 == 0:
                    bar = ax.bar(x, means_dict[algo], width, label=algo)
                elif i % 3 == 1:
                    bar = ax.bar(x - (width * sides), means_dict[algo], width, label=algo)
                else:
                    bar = ax.bar(x + (width * sides), means_dict[algo], width, label=algo)
                    sides += 1
                ax.bar_label(bar, padding=3)
                i += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(metric)
        ax.set_xticks(x, groups_label)
        ax.legend()
        plt.xticks(rotation=15)
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), fancybox=True, shadow=True, ncol=3)
        fig.tight_layout()
        # Pasta para salvar a figura
        file_dir = PathDirFile.set_graphics_file(dataset_name, metric + '.png')
        # Salvar figura no disco
        plt.savefig(
            file_dir,
            format='png',
            dpi=400,
            bbox_inches='tight'
        )
        # Figura fechada
        plt.close('all')
