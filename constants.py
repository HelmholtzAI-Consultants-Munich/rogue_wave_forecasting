########################################################
# Constants
########################################################

# Data paths
from pathlib import Path

import numpy as np

DIR_ROOT = Path(__file__).resolve().parent  # constants.py sits in the repo root
DIR_DATA = DIR_ROOT / "data"
DIR_RESULTS = DIR_ROOT / "results"

FILE_DATA_RAW = DIR_DATA / "abin_matrix_full_encoded_new.csv"
FILE_DATA_PROCESSED = DIR_DATA / "data_train_test.pickle"

FILE_CV_RESULTS = "cv_results.csv"
FILE_MODEL_AND_DATA = "model_and_data.pkl"
FILE_PERFORMANCE_TRAIN_PLOT = "performance_train.png"
FILE_PERFORMANCE_TRAIN_CSV = "performance_train.csv"
FILE_PERFORMANCE_TEST_PLOT = "performance_test.png"
FILE_PERFORMANCE_TEST_CSV = "performance_test.csv"
FILE_SHAP_TRAIN = "shap_train.pkl"
FILE_SHAP_TEST = "shap_test.pkl"

# Data processing

COLUMN_TARGET = "AI_10min"  # Abnormality Index (AbnI) = H_max / H_s over the following 10 minutes
ROGUE_WAVE_THRESHOLD = 2.0  # rogue-wave definition, Haver & Andersen (2000)
WAVE_CUTOFF = 2.7  # cutoff for AbnI, used to filter out physically non-meaningful waves

# The 17 metocean features used for modelling
COLUMN_FEATURES = [
    "H_s",
    "lambda_40",
    "lambda_30",
    "L_deep",
    "s",
    "mu",
    "kh",
    "T_p",
    "nu",
    "Q_p",
    "BFI",
    "r",
    "v_wind",
    "v_gust",
    "T_air",
    "p",
    "Delta_p_1h",
]
COLUMN_FOLD = "fold"

# Plotting

TIME_SERIES_PLOT_SUBSAMPLE = 10_000  # subsample for the time series plot

ROLLING_WINDOW_SMALL = 100  # window size for the small rolling mean
ROLLING_WINDOW_MEDIUM = 500  # window size for the medium rolling mean
ROLLING_WINDOW_LARGE = 5000  # window size for the large rolling mean

N_BLOCKS = 8  # coarse blocks of the record used to colour the embeddings
EMBED_SUBSAMPLE = 50_000  # systematic sub-sample used for UMAP and for the scatter panels

# Model training

TEST_FRACTION = 0.20
NUM_CV = 5
PURGE_GAP = 1_000


HYPERPARAMETER_GRID_LM = {
    "alpha": np.logspace(-5, 0, 9),  # Overall Elastic Net regularisation strength
    "l1_ratio": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0],  # L1–L2 penalty mixture
    "max_iter": [20_000],  # Maximum coordinate-descent iterations
    "tol": [1e-4],  # Convergence tolerance
    "selection": ["cyclic"],  # Update coefficients sequentially
}

HYPERPARAMETER_GRID_SVM = {
    "kernel": ["rbf"],  # Nonlinear radial-basis-function kernel
    "C": [0.3, 3.0, 30.0],  # Penalty for prediction errors
    "gamma": [0.03, 0.10],  # Locality of each training sample's influence
    "epsilon": [0.05, 0.15],  # Width of the error-insensitive regression tube
}

HYPERPARAMETER_GRID_RF = {
    "n_estimators": [100, 500],  # Number of trees
    "max_depth": [None, 10, 20, 30],  # Maximum depth of each tree
    "max_samples": [0.25, 0.50, 0.75],  # Fraction of rows sampled per tree
    "max_features": ["sqrt", 0.5, 1.0],  # Features considered at each split
    "min_samples_leaf": [20, 50, 100],  # Minimum observations in a leaf
    "min_samples_split": [2, 10, 50],  # Minimum observations required to split
    "criterion": ["squared_error"],  # Split quality based on variance reduction
}

HYPERPARAMETER_GRID_XGB = {
    "n_estimators": [100, 500],  # Number of boosting trees
    "learning_rate": [0.02, 0.10],  # Contribution of each new tree
    "max_depth": [5, 10, 20],  # Maximum interaction depth of each tree
    "min_child_weight": [1, 20],  # Minimum Hessian weight in a child
    "subsample": [0.25, 0.50, 0.75],  # Fraction of rows used per tree
    "colsample_bytree": [0.8],  # Fraction of features used per tree
    "gamma": [0.0, 0.5],  # Minimum gain required for a split
    "reg_alpha": [0.01, 0.50, 1.0],  # L1 regularisation on leaf weights
    "reg_lambda": [5.0],  # L2 regularisation on leaf weights
    "tree_method": ["hist"],  # Histogram-based tree construction
}
