from datasets.charts import DatasetChart
from datasets.registred_datasets import RegisteredDataset

for dataset_name in RegisteredDataset.DATASET_LIST:
    print(dataset_name)
    dt_chat = DatasetChart(dataset_name)
    dt_chat.item_long_tail()
    dt_chat.genres()
