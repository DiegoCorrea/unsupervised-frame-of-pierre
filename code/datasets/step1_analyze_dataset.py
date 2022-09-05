from datasets.registred_datasets import RegisteredDataset

for dataset_name in RegisteredDataset.DATASET_LIST:
    print(dataset_name)
    dataset = RegisteredDataset.load_dataset(dataset_name)
    dataset.raw_data_basic_info()
    dataset.clean_data_basic_info()
