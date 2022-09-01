import multiprocessing
from settings.utils import get_size

import os


class Constants:
    """
    Constants class.
    Responsible for leading with all constant values used by the system
    """

    # Machine config #
    MEM_RAM = get_size()
    N_CORES = multiprocessing.cpu_count()
    os.environ['NUMEXPR_MAX_THREADS'] = str(N_CORES)

    # Experimental #
    K_FOLDS_VALUE = 5
    N_TRIAL_VALUE = 7
    PROFILE_LEN_CUT_VALUE = K_FOLDS_VALUE

    # SEARCH
    N_INTER = 127

    # Algorithm hyper param #
    ALPHA_VALUE = 0.01

    # Data Model config #
    RECOMMENDATION_LIST_SIZE = 10
    CANDIDATES_LIST_SIZE = 100

    # Minkowski Distance param
    DIMENSION_VALUE = 3
