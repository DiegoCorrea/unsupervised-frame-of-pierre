from evaluations.best_worst import best_and_worst_systems, best_and_worst_fairness_measure
from evaluations.hypothesis import welch
from graphics.research_questions.perspectives import components_box_graphic, fairness_box_graphic
from settings.path_dir_file import PathDirFile
import pandas as pd

from utils.read_from_terminal import read_input_chart_analises


def charts(setup_config):
    for dataset_name in setup_config['dataset']:
        data = pd.read_csv(PathDirFile.get_decision_file(dataset_name))

        # Ranking
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MAP")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MAP")
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRR")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRR")

        # Calibration
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MACE")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MACE")
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRMC")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="MRMC")

        # Coefficients
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CCE")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CCE")
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CMC")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="CMC")

        # Coefficients
        components_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="PERFORMANCE")
        fairness_box_graphic(data=data, dataset_name=dataset_name, order_by_metric="PERFORMANCE")


def analyses(setup_config):
    for dataset_name in setup_config['dataset']:
        print("|" * 100)
        print("-"*10, " ", dataset_name, " ", "-"*10)
        print("|" * 100)

        results = pd.read_csv(PathDirFile.get_decision_file(dataset_name))
        results.fillna(0.0)

        # execution_time_analyze(data=results)
        best_and_worst_systems(data=results, order_by_metric='MAP', ascending=False)
        best_and_worst_systems(data=results, order_by_metric='MRR', ascending=False)
        best_and_worst_systems(data=results, order_by_metric='MACE', ascending=True)
        best_and_worst_systems(data=results, order_by_metric='CCE', ascending=True)

        best_and_worst_fairness_measure(data=results, order_by_metric='MAP', ascending=False)
        best_and_worst_fairness_measure(data=results, order_by_metric='MRR', ascending=False)
        best_and_worst_fairness_measure(data=results, order_by_metric='MACE', ascending=True)
        best_and_worst_fairness_measure(data=results, order_by_metric='CCE', ascending=True)

        welch(data=results, order_by_metric='MAP', ascending=False)
        welch(data=results, order_by_metric='MRR', ascending=False)
        welch(data=results, order_by_metric='MACE', ascending=True)
        welch(data=results, order_by_metric='CCE', ascending=True)

        print("|" * 100)


def main():
    setup_config = read_input_chart_analises()
    if setup_config['opt'] == "CHART":
        charts(setup_config)
    elif setup_config['opt'] == "ANALYZE":
        analyses(setup_config)
    else:
        pass


if __name__ == '__main__':
    """
    Starting the chart and analysis creation
    """
    main()
