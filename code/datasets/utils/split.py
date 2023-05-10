import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from settings.constants import Constants
from settings.labels import Label

# Shared memory between the processes
global_train = []
global_test = []


def user_split_in_kfolds(
        user_transactions: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE, n_folds: int = Constants.K_FOLDS_VALUE
) -> None:
    """
    Split the user transaction in K-folds.
    :param user_transactions: A Pandas DataFrame with user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int representing a number of the k folds.
    """
    user_transactions.reset_index(inplace=True)
    kf = KFold(n_splits=n_folds, random_state=42 * trial * n_folds, shuffle=True)

    i = 0
    for train, test in kf.split(user_transactions):
        global_train[i].append(user_transactions.iloc[train])
        global_test[i].append(user_transactions.iloc[test])
        i += 1


def concat_folds(train_results_df, test_results_df):
    """
    To concat the users' folds on the same structure.
    """
    return pd.concat(train_results_df, sort=False), pd.concat(test_results_df, sort=False)


def compute_kfold(
        transactions_df: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE,
        n_folds: int = Constants.K_FOLDS_VALUE, n_jobs: int = Constants.N_CORES
) -> list:
    """
    Prepare the users to be processed in parallel with the joblib.
    :param transactions_df: A Pandas DataFrame with user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int representing a number of the k folds.
    :param n_jobs: An int representing the number of cores used to parallel the operation.
    :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
    """
    # Preparing: users, results dataframe and shared queue over processes

    grouped_transactions = transactions_df.groupby(by=[Label.USER_ID])

    delayed_list = (
        delayed(user_split_in_kfolds)(transactions, trial, n_folds)
        for user_id, transactions in grouped_transactions
    )
    out = Parallel(n_jobs=n_jobs, verbose=10, batch_size=128, require='sharedmem')(delayed_list)

    # delayed_concat = (
    #     delayed(concat_folds)(train_df, test_df)
    #     for train_df, test_df in zip(global_train, global_test)
    # )
    # resp = list(Parallel(
    #     n_jobs=n_jobs, verbose=10, batch_size=64, prefer='processes', backend='multiprocessing'
    # )(delayed_concat))

    resp = []
    for train_df, test_df in zip(global_train, global_test):
        resp.append(concat_folds(train_df, test_df))
    return resp


def split_with_joblib(
        transactions_df: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE, n_folds: int = Constants.K_FOLDS_VALUE
) -> list:
    """
    Prepare the users to be processed in parallel with the joblib.
    :param transactions_df: A Pandas DataFrame with user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int representing a number of the k folds.
    :return: A list composed of the fold in positions, each fold position has [0] as the k fold train transactions and [1] as the k fold test transactions.
    """
    global global_train
    global global_test

    global_train = [[] for _ in range(n_folds)]
    global_test = [[] for _ in range(n_folds)]

    resp = compute_kfold(transactions_df=transactions_df, trial=trial, n_folds=n_folds)

    return resp
