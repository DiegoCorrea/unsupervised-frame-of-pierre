class Label:
    # Data labels
    USER_ID = 'USER_ID'
    ITEM_ID = 'ITEM_ID'
    TRANSACTION_VALUE = 'TRANSACTION_VALUE'
    PREDICTED_VALUE = 'PREDICTED_VALUE'
    GENRES = 'GENRES'
    TITLE = 'TITLE'
    TIME = 'TIMESTAMP'
    ORDER = 'ORDER'
    BIAS_VALUE = 'BIAS_VALUE'

    ########################################################
    # Recommenders labels
    USER_KNN_BASIC = 'USER_KNN_BASIC'
    ITEM_KNN_BASIC = 'ITEM_KNN_BASIC'
    SVD = 'SVD'
    SVDpp = 'SVDpp'
    NMF = 'NMF'
    SLOPE = 'SLOPE'
    CO_CLUSTERING = 'CO_CLUSTERING'

    DEFAULT_REC = SVD

    SURPRISE_RECOMMENDERS = [
        NMF,
        CO_CLUSTERING,
        SLOPE,
        SVD,
        SVDpp,
        USER_KNN_BASIC,
        ITEM_KNN_BASIC
    ]
    REGISTERED_RECOMMENDERS = SURPRISE_RECOMMENDERS

    ########################################################
    # Evaluation Metric labels
    # # Ranking Metrics
    MAP = 'MAP'
    MRR = 'MRR'
    # # Calibration Metrics
    MACE = 'MACE'
    MRMC = 'MRMC'

    ########################################################
    # Post-processing labels #

    # # Tradeoff Balances
    LIN_TRADEOFF = 'LIN'
    LOG_TRADEOFF = 'LOG'
    ACCESSIBLE_TRADEOFF_LIST = [
        LIN_TRADEOFF,
        LOG_TRADEOFF
    ]
    DEFAULT_TRADEOFF = LIN_TRADEOFF

    # # Relevance Measures
    SUM_RELEVANCE = 'SUM'
    NDCG_RELEVANCE = "NDCG"
    ACCESSIBLE_RELEVANCE_LIST = [
        SUM_RELEVANCE,
        NDCG_RELEVANCE
    ]
    DEFAULT_RELEVANCE = SUM_RELEVANCE

    # # Distributions
    DISTRIBUTION_LABEL = 'DISTRIBUTION_LABEL'
    # # # Genre Distributions
    CLASS_WEIGHTED_STRATEGY = "CWS"
    WEIGHTED_PROBABILITY_STRATEGY = "WPS"
    ACCESSIBLE_DISTRIBUTION_LIST = [
        # Genre Distributions
        CLASS_WEIGHTED_STRATEGY,
        WEIGHTED_PROBABILITY_STRATEGY
    ]
    DEFAULT_DISTRIBUTION = CLASS_WEIGHTED_STRATEGY

    # # Selector Algorithms
    SELECTOR_LABEL = 'SELECTOR_LABEL'
    SURROGATE_SELECTOR = 'SURROGATE'
    ACCESSIBLE_SELECTOR_LIST = [
        SURROGATE_SELECTOR
    ]
    DEFAULT_SELECTOR = SURROGATE_SELECTOR

    # # Tradeoff Weights
    TRADEOFF_WEIGHT_LABEL = 'TRADEOFF_WEIGHT'
    CGR_WEIGHT = "CGR"
    VAR_WEIGHT = "VAR"
    PERSON_WEIGHT = [
        VAR_WEIGHT,
        CGR_WEIGHT,
    ]
    CONST_WEIGHT = [
        "C@0.0", "C@0.1", "C@0.2", "C@0.3", "C@0.4", "C@0.5",
        "C@0.6", "C@0.7", "C@0.8", "C@0.9", "C@1.0"
    ]
    ACCESSIBLE_WEIGHT_LIST = PERSON_WEIGHT + CONST_WEIGHT
    DEFAULT_WEIGHT = VAR_WEIGHT

    # # Calibration Measures
    CALIBRATION_MEASURE_LABEL = 'CALIBRATION_MEASURE'
    # # # Minkowski Family
    MINKOWSKI = 'MINKOWSKI'
    CHEBYSHEV = 'CHEBYSHEV'
    EUCLIDEAN = 'EUCLIDEAN'
    CITY_BLOCK = 'CITY_BLOCK'
    # # # L1 Family
    SORENSEN = "SORESEN"
    GOWER = "GOWER"
    SOERGEL = "SOERGEL"
    KULCZYNSKI_D = "KULCZYNSKI_D"
    CANBERRA = "CANBERRA"
    LORENTZIAN = "LORENTZIAN"
    # # # Intersection Family
    INTERSECTION_SIM = "INTERSECTION_SIM"  # SIMILARITY
    INTERSECTION_DIV = "INTERSECTION_DIV"
    WAVE_HEDGES = "WAVE"
    CZEKANOWSKI_SIM = "CZEKANOWSKI_SIM"  # SIMILARITY
    CZEKANOWSKI_DIV = "CZEKANOWSKI_DIV"
    MOTYKA_SIM = "MOTYKA_SIM"  # SIMILARITY
    MOTYKA_DIV = "MOTYKA_DIV"
    KULCZYNSKI_S = "KULCZYNSKI_S"  # SIMILARITY
    RUZICKA = "RUZICKA"  # SIMILARITY
    TANIMOTO = "TONIMOTO"
    # # # Inner Production Family
    INNER = "INNER"  # SIMILARITY
    HARMONIC = "HARMONIC"  # SIMILARITY
    COSINE = "COSINE"  # SIMILARITY
    KUMAR = "KUMAR_HASSEBROOK"  # SIMILARITY
    JACCARD = "JACCARD"
    DICE_SIM = "DICE_SIM"  # SIMILARITY
    DICE_DIV = "DICE_DIV"
    # # # Fidelity Family
    FIDELITY = 'FIDELITY'  # SIMILARITY
    BHATTACHARYYA = "BHATTACHARYYA"
    HELLINGER = "HELLINGER"
    MATUSITA = "MATUSITA"
    SQUARED_CHORD_SIM = "SQUARED_CHORD_SIM"  # SIMILARITY
    SQUARED_CHORD_DIV = "SQUARED_CHORD_DIV"
    # # # Chi Square Family
    SQUARED_EUCLIDEAN = "SQUARED_EUCLIDEAN"
    CHI = "CHI_SQUARE"
    NEYMAN = "NEYMAN"
    SQUARED = "SQUARED_CHI"
    PROBABILISTIC = "PROBABILISTIC_CHI"
    DIVERGENCE = "DIVERGENCE"
    CLARK = "CLARK"
    ADDITIVE = "ADDITIVE_CHI"
    # # # Shannon's Entropy Family
    KULLBACK = "KL"
    JEFFREYS = "JEFFREYS"
    K_DIVERGENCE = "K_DIV"
    TOPSOE = "TOPSOE"
    JENSEN_SHANNON = "JENSEN_SHANNON"
    JENSEN_DIFF = "JENSEN_DIFF"
    # # # Combinations
    TANEJA = "TANEJA"
    KUMAR_JOHNSON = "KUMAR_JOHNSON"
    AVG = "AVG"
    WTV = "WTV"
    # # # Vicissitude
    VICIS_WAVE = "VICIS_WAVE"
    VICIS_EMANON2 = "VICIS_EMANON2"
    VICIS_EMANON3 = "VICIS_EMANON3"
    VICIS_EMANON4 = "VICIS_EMANON4"
    VICIS_EMANON5 = "VICIS_EMANON5"
    VICIS_EMANON6 = "VICIS_EMANON6"
    # # #
    DIVERGENCE_LIST = [
        # Minkowski Family
        MINKOWSKI,
        EUCLIDEAN,
        CITY_BLOCK,
        CHEBYSHEV,
        # L1 Family
        SORENSEN,
        GOWER,
        SOERGEL,
        KULCZYNSKI_D,
        CANBERRA,
        LORENTZIAN,
        # Intersection Family
        INTERSECTION_DIV,
        WAVE_HEDGES,
        CZEKANOWSKI_DIV,
        MOTYKA_DIV,
        TANIMOTO,
        # Inner Production Family
        JACCARD,
        DICE_DIV,
        # Fidelity Family
        BHATTACHARYYA,
        HELLINGER,
        MATUSITA,
        SQUARED_CHORD_DIV,
        # Chi Square Family
        SQUARED_EUCLIDEAN,
        CHI,
        NEYMAN,
        SQUARED,
        PROBABILISTIC,
        DIVERGENCE,
        CLARK,
        ADDITIVE,
        # Shannon's Entropy Family
        KULLBACK,
        JEFFREYS,
        K_DIVERGENCE,
        TOPSOE,
        JENSEN_SHANNON,
        JENSEN_DIFF,
        # Combinations
        TANEJA,
        KUMAR_JOHNSON,
        AVG,
        WTV,
        # Vicissitude
        VICIS_WAVE,
        VICIS_EMANON2,
        VICIS_EMANON3,
        VICIS_EMANON4,
        VICIS_EMANON5,
        VICIS_EMANON6,
    ]
    SIMILARITY_LIST = [
        # Intersection
        INTERSECTION_SIM,
        CZEKANOWSKI_SIM,
        MOTYKA_SIM,
        KULCZYNSKI_S,
        RUZICKA,
        # Inner Production Family
        INNER,
        HARMONIC,
        COSINE,
        KUMAR,
        DICE_SIM,
        # Fidelity Family
        FIDELITY,
        SQUARED_CHORD_SIM,
    ]
    ACCESSIBLE_CALIBRATION_LIST = DIVERGENCE_LIST + SIMILARITY_LIST
    DEFAULT_CALIBRATION = CHI

    ########################################################
    #
    NUMBER_USER = "#_USERS"
    NUMBER_ITEMS = "#_ITEMS"
    TOTAL_TIMES = 'TOTAL_TIMES'
    USER_MODEL_SIZE_LABEL = 'user_model_size'
    NUMBER_OF_SHORT_TAIL_ITEMS_LABEL = '#_of_short_tail_items'
    NUMBER_OF_MEDIUM_TAIL_ITEMS_LABEL = '#_of_medium_tail_items'
    NUMBER_OF_LONG_TAIL_ITEMS_LABEL = '#_of_long_tail_items'
    PERCENTAGE_OF_SHORT_TAIL_ITEMS_LABEL = '%_of_short_tail_items'
    PERCENTAGE_OF_MEDIUM_TAIL_ITEMS_LABEL = '%_of_medium_tail_items'
    PERCENTAGE_OF_LONG_TAIL_ITEMS_LABEL = '%_of_long_tail_items'
    NUMBER_OF_CLASSES = "#_CLASSES"
    group_of_users = "USER_GROUP"

    # Popularity types
    TYPE_OF_POPULARITY = 'TYPE_OF_POPULARITY'
    SHORT_TAIL_TYPE = 'short_head'
    MEDIUM_TAIL_TYPE = 'medium_tail'
    LONG_TAIL_TYPE = 'long_tail'

    NICHE_TYPE = 'niche'
    DIVERSE_TYPE = 'diverse'
    FOCUSED_TYPE = 'focused'

    #
    TYPE_OF_GENRE_GROUPS_LABEL = 'TYPE_OF_GENRE_GROUPS'
    EXPLORER_TYPE = 'explorer'
    COMMON_TYPE = 'common'
    LOYAL_TYPE = 'loyal'

    reco_popularity_label = "reco_popularity"
    normalized_reco_popu_label = 'normalized_reco_popularity'
    preference_popularity_label = "preference_popularity"
    normalized_preference_popu_label = 'normalized_preference_popularity'

    # ######################### #
    #    Evaluate Variables     #
    # ######################### #
    # Structure Labels

    # List of Strings

    FIXED_LABEL = 'FIXED'
    LAMBDA_LABEL = 'LAMBDA'
    LAMBDA_VALUE_LABEL = 'LAMBDA_VALUE'
    EVALUATION_METRIC_LABEL = 'EVALUATION_METRIC'
    EVALUATION_VALUE_LABEL = 'EVALUATION_VALUE'

    # ####################################### #
    #    Config Post Processing Variables     #
    # ####################################### #

    GREEDY_ALGORITHM_LABEL = 'GREEDY_ALGORITHM'
    SURROGATE_LABEL = 'SURROGATE'

    TRADE_OFF_LABEL = 'TRADE_OFF'
    COUNT_GENRES_TRADE_OFF_LABEL = 'COUNT_GENRES'
    SUN_GENRES_PROBABILITY_TRADE_OFF_LABEL = 'SUN_GENRES_PROBABILITY'
    VARIANCE_TRADE_OFF_LABEL = 'VARIANCE'
    MANUAL_VALUE_LABEL = 'MANUAL_VALUE'

    CALIBRATION_LABEL = 'CALIBRATION'

    FIXED_LAMBDA_LABEL = 'FIXED_LAMBDA_LABEL'
    PERSON_LAMBDA_LABEL = 'PERSON_LAMBDA_LABEL'
    #
    # EVALUATION_LIST_LABELS = [ALGORITHM_LABEL, CALIBRATION_LABEL, FAIRNESS_METRIC_LABEL, LAMBDA_LABEL,
    # LAMBDA_VALUE_LABEL, EVALUATION_METRIC_LABEL, EVALUATION_VALUE_LABEL]
