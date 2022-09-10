from scipy.stats import randint, uniform


class SurpriseParams:
    # 1: User KNN
    USER_KNN_SEARCH_PARAMS = {"k": randint(3, 101),
                              "sim_options": {'name': ['pearson', 'cosine', 'msd'], 'user_based': [True]}}
    # 2: Item KNN
    ITEM_KNN_SEARCH_PARAMS = {"k": randint(3, 101),
                              "sim_options": {'name': ['pearson', 'cosine', 'msd'], 'user_based': [False]}}
    # 3: SVD
    SVD_SEARCH_PARAMS = {"n_factors": randint(10, 150), "n_epochs": randint(10, 150), "lr_all": uniform(0.001, 0.01),
                         "reg_all": uniform(0.01, 0.1)}
    # 4: SVD++
    SVDpp_SEARCH_PARAMS = {"n_factors": randint(10, 150), "n_epochs": randint(10, 150), "lr_all": uniform(0.001, 0.01),
                           "reg_all": uniform(0.01, 0.1)}
    # 5: NMF
    NMF_SEARCH_PARAMS = {"n_factors": randint(10, 150), "n_epochs": randint(10, 150), "reg_pu": uniform(0.01, 0.1),
                         "reg_qi": uniform(0.01, 0.1), "reg_bu": uniform(0.01, 0.1), "reg_bi": uniform(0.01, 0.1),
                         "lr_bu": uniform(0.001, 0.01), "lr_bi": uniform(0.001, 0.01), "biased": [True]}
    # 6: Co Clustering
    CLUSTERING_SEARCH_PARAMS = {"n_cltr_u": randint(3, 11), "n_cltr_i": randint(3, 11), "n_epochs": randint(10, 150)}


class ConformityParams:
    # 1: Agglomerative Clustering
    AGGLOMERATIVE_CLUSTERING_SEARCH_PARAMS = {

    }

    # 2: Bernoulli Restricted Boltzmann Machine (RBM).
    BERNOULLI_RBM_SEARCH_PARAMS = {

    }

    # 3: K-Means clustering.
    KMEANS_SEARCH_PARAMS = {

    }
