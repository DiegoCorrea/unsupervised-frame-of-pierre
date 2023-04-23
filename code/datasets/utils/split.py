import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from settings.constants import Constants
from settings.labels import Label


def user_split_in_kfolds(user_transactions: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE,
                         n_folds: int = Constants.K_FOLDS_VALUE) -> tuple:
    """
    Split the user transaction in K-folds.
    :param user_transactions: A Pandas DataFrame with a user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int that represents a number of the k folds.
    :return: A tuple with two positions, [0] is the k fold train transactions and [1] is the k fold test transactions.
    """
    user_transactions.reset_index(inplace=True)
    kf = KFold(n_splits=n_folds, random_state=42 * trial * n_folds, shuffle=True)
    train_fold = []
    test_fold = []
    for train, test in kf.split(user_transactions):
        train_fold.append(user_transactions.iloc[train])
        test_fold.append(user_transactions.iloc[test])
    return (train_fold, test_fold)


def compute_kfold(transactions_df: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE,
                  n_folds: int = Constants.K_FOLDS_VALUE, n_jobs: int = Constants.N_CORES):
    """
    Prepare the users to be processed in parallel with the joblib.
    :param transactions_df: A Pandas DataFrame with a user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int that represents a number of the k folds.
    :param n_jobs: An int that represents a number of cores used to parallel the operation.
    :return: A tuple with two positions, [0] is the k fold train transactions and [1] is the k fold test transactions.
    Each position has a list with n_folds positions.
    """
    # Preparing: users, results dataframe and shared queue over processes
    # users_ids = transactions_df[Label.USER_ID].unique().tolist()
    grouped_transactions = transactions_df.groupby(by=[Label.USER_ID])

    delayed_list = (
        delayed(user_split_in_kfolds)(transactions, trial, n_folds)
        for user_id, transactions in grouped_transactions
    )
    out = Parallel(n_jobs=n_jobs, verbose=10, batch_size=64)(delayed_list)
    return out


def vectorize_kfold(out, n_folds: int = Constants.K_FOLDS_VALUE):
    train_results_list = [[] for _ in range(n_folds)]
    test_results_list = [[] for _ in range(n_folds)]
    for result in out:
        train_side = result[0]
        test_side = result[1]
        for i in range(n_folds):
            train_results_list[i].append(train_side[i])
            test_results_list[i].append(test_side[i])
    return train_results_list, test_results_list


def split_with_joblib(transactions_df: pd.DataFrame, trial: int = Constants.N_TRIAL_VALUE,
                      n_folds: int = Constants.K_FOLDS_VALUE) -> tuple:
    """
    Prepare the users to be processed in parallel with the joblib.
    :param transactions_df: A Pandas DataFrame with a user transactions.
    :param trial: An int that represents a number of the experimental trial.
    :param n_folds: An int that represents a number of the k folds.
    :return: A tuple with two positions, [0] is the k fold train transactions and [1] is the k fold test transactions.
    Each position has a list with n_folds positions.
    """
    out = compute_kfold(transactions_df=transactions_df, trial=trial, n_folds=n_folds)
    train_results_list, test_results_list = vectorize_kfold(out=out, n_folds=n_folds)
    # Concat and resume the results

    train_results_df = [pd.DataFrame() for _ in range(n_folds)]
    test_results_df = [pd.DataFrame() for _ in range(n_folds)]
    for i in range(n_folds):
        train_results_df[i] = pd.concat(train_results_list[i], sort=False)
        test_results_df[i] = pd.concat(test_results_list[i], sort=False)

    # train_results_df = [pd.DataFrame() for _ in range(n_folds)]
    # test_results_df = [pd.DataFrame() for _ in range(n_folds)]
    # for result in out:
    #     train_side = result[0]
    #     test_side = result[1]
    #     for i in range(n_folds):
    #         train_results_df[i] = pd.concat([train_results_df[i], train_side[i]], sort=False)
    #         test_results_df[i] = pd.concat([test_results_df[i], test_side[i]], sort=False)
    return (train_results_df, test_results_df)
