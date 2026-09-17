############################################
# imports
############################################

import os
import sys
import time
import pickle
import argparse

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

sys.path.append("./")
sys.path.append("../scripts/")
import utils

############################################
# Constants
############################################

COLUMN_TARGET = "AI_10min"
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

FILE_CV_RESULTS = "cv_results.csv"
FILE_MODEL_AND_DATA = "model_and_data.pkl"
FILE_PERFORMANCE_TEST = "performance_test"
FILE_PERFORMANCE_TRAIN = "performance_train"
FILE_MODEL_SIZE = "model_size.pickle"

############################################
# Train Model
############################################


def argument_parser():
    parser = argparse.ArgumentParser(description="Run SHAP.")
    parser.add_argument("--model_type", type=str, help="Type of model to train")
    parser.add_argument("--file_data", type=str, help="Path to the data file")
    parser.add_argument("--dir_output", type=str, help="Directory for output files")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")

    args = parser.parse_args()
    model_type = args.model_type
    file_data = args.file_data
    dir_output = args.dir_output
    n_jobs = args.n_jobs

    return model_type, file_data, dir_output, n_jobs


def get_hyperparameter_grid(model_type):
    if model_type == "lm":
        hyperparameter_grid = {
            "alpha": np.logspace(-5, 0, 9),  # Overall Elastic Net regularisation strength
            "l1_ratio": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0],  # L1–L2 penalty mixture
            "max_iter": [20_000],  # Maximum coordinate-descent iterations
            "tol": [1e-4],  # Convergence tolerance
            "selection": ["cyclic"],  # Update coefficients sequentially
        }

    elif model_type == "svm":
        hyperparameter_grid = {
            "kernel": ["rbf"],  # Nonlinear radial-basis-function kernel
            "C": [0.3, 3.0, 30.0],  # Penalty for prediction errors
            "gamma": [0.03, 0.10],  # Locality of each training sample's influence
            "epsilon": [0.05, 0.15],  # Width of the error-insensitive regression tube
        }

    elif model_type == "rf":
        hyperparameter_grid = {
            "n_estimators": [100, 500],  # Number of trees
            "max_depth": [None, 10, 20, 30],  # Maximum depth of each tree
            "max_samples": [0.25, 0.50, 0.75],  # Fraction of rows sampled per tree
            "max_features": ["sqrt", 0.5, 1.0],  # Features considered at each split
            "min_samples_leaf": [2, 5, 20, 50, 100],  # Minimum observations in a leaf
            "min_samples_split": [2, 10, 50],  # Minimum observations required to split
            "criterion": ["squared_error"],  # Split quality based on variance reduction
        }

    elif model_type == "xgb":
        hyperparameter_grid = {
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return hyperparameter_grid


def get_model_instance(model_type, seed):
    if model_type == "lm":
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(random_state=seed)

    elif model_type == "svm":
        from thundersvm import SVR

        model = SVR()

    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=seed)

    elif model_type == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(random_state=seed, n_jobs=1)  # n_jobs=1 to avoid nested parallelism

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def load_data(file_data):

    with open(file_data, "rb") as handle:
        data_train, data_test = pickle.load(handle)

    X_train, y_train = data_train[COLUMN_FEATURES], data_train[COLUMN_TARGET]
    X_test, y_test = data_test[COLUMN_FEATURES], data_test[COLUMN_TARGET]
    cv_groups = data_train[COLUMN_FOLD].to_numpy()

    return data_train, data_test, X_train, y_train, X_test, y_test, cv_groups


def run_CV(model, hyperparameter_grid, num_cv, X, y, groups, n_jobs, verbose=0):

    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

    # Tune hyperparameters with grouped CV: each chronological fold validates exactly once
    skf = list(GroupKFold(n_splits=num_cv).split(X, y, groups=groups))

    gridsearch_cv = GridSearchCV(
        model,
        hyperparameter_grid,
        cv=skf,
        n_jobs=n_jobs,
        scoring=scoring,
        refit="rmse",
        verbose=verbose,
    )
    gridsearch_cv.fit(X, y)

    # Take the best estimator
    model = gridsearch_cv.best_estimator_

    # Collect CV Results (multi-metric scoring → mean_test_<name>, not mean_test_score)
    cv_results = pd.concat(
        [
            pd.DataFrame(gridsearch_cv.cv_results_["params"]),
            pd.DataFrame(
                {
                    "rmse": gridsearch_cv.cv_results_["mean_test_rmse"],
                    "mae": gridsearch_cv.cv_results_["mean_test_mae"],
                    "r2": gridsearch_cv.cv_results_["mean_test_r2"],
                }
            ),
        ],
        axis=1,
    )

    return model, cv_results


def evaluate_model(model, X, y, set_name, dir_output, filename, plot=True, save=False):
    # Predict labels
    y_pred = model.predict(X)

    mse = round(mean_squared_error(y, y_pred), 3)
    mae = round(mean_absolute_error(y, y_pred), 3)
    r2 = round(r2_score(y, y_pred), 3)
    spearman_r = round(spearmanr(y, y_pred).correlation, 3)

    if plot:
        print(f"Evaluate on {set_name} Set")
        textstr = f"$MSE={mse}$\n$MAE={mae}$\n$R^2={r2}$\n$Spearman\\ R={spearman_r}$"
        fig, ax = utils.plot_predictions(y_true=y, y_pred=y_pred, textstr=textstr)
        fig.savefig(f"{dir_output}/{filename}.png", bbox_inches="tight", dpi=300)

    if save:
        output = [y, y_pred, mse, mae, r2, spearman_r]
        with open(f"{dir_output}/{filename}.pkl", "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"{set_name} set performance: r2={r2:.3f}, spearman_r={spearman_r:.3f}, mse={mse:.3f}, mae={mae:.3f}"
    )


def store_predictions(model, X_train, y_train, X_test, y_test, dir_output):

    set_name = "Training"
    evaluate_model(
        model, X_train, y_train, set_name, dir_output, FILE_PERFORMANCE_TRAIN, plot=False, save=True
    )
    set_name = "Test"
    evaluate_model(model, X_test, y_test, set_name, dir_output, FILE_PERFORMANCE_TEST, plot=False, save=True)


def get_model_size(model):
    serialized = pickle.dumps(
        model,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    size_gib = len(serialized) / 1024**3
    print(f"Serialized model size: {size_gib:.4f} GiB")

    return size_gib


def train(model_type, file_data, dir_output, n_jobs):

    print(f"Setup {model_type}...")
    seed = 42
    num_cv = 5
    hyperparameter_grid = get_hyperparameter_grid(model_type)

    print(f"Using {n_jobs} cores from {os.cpu_count()} available cores.")
    print(hyperparameter_grid)

    print("Loading data...")
    data_train, data_test, X_train, y_train, X_test, y_test, cv_groups = load_data(file_data)

    print(
        f"{len(X_train):,} train rows in folds {sorted(set(cv_groups))}, "
        f"{len(X_test):,} test rows, {X_train.shape[1]} features."
    )

    print("Getting model instance...")
    base_model = get_model_instance(model_type, seed)
    needs_scaling = model_type in {"lm", "svm"}
    pipeline = Pipeline(
        [("scale", StandardScaler() if needs_scaling else "passthrough"), ("model", base_model)]
    )
    hyperparameter_grid = {f"model__{name}": values for name, values in hyperparameter_grid.items()}

    print("Tuning hyperparameters with cross-validation...")
    start = time.time()
    model, cv_results = run_CV(
        pipeline, hyperparameter_grid, num_cv, X_train, y_train, cv_groups, n_jobs, verbose=2
    )
    end = time.time()
    print(f"Model training took {end - start:.2f} seconds")

    print("Evaluating model parameter configurations...")
    cv_results = cv_results.sort_values("score", ascending=False).reset_index(drop=True)
    cv_results.to_csv(f"{dir_output}/{FILE_CV_RESULTS}", index=False)
    print(cv_results)

    if model_type == "svm":
        store_predictions(model, X_train, y_train, X_test, y_test, dir_output)

    with open(f"{dir_output}/{FILE_MODEL_AND_DATA}", "wb") as handle:
        pickle.dump([data_train, data_test, model], handle, protocol=pickle.HIGHEST_PROTOCOL)

    size_gib = get_model_size(model)

    print("Done.")


def load_data_and_model(file_data_model, dir_output, output=True):

    # Load and unpack the data
    with open(file_data_model, "rb") as handle:
        data_train, data_test, model = pickle.load(handle)

    X_train, y_train = data_train[COLUMN_FEATURES], data_train[COLUMN_TARGET]
    X_test, y_test = data_test[COLUMN_FEATURES], data_test[COLUMN_TARGET]

    if isinstance(model, RandomForestRegressor):
        tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
        average_depth = sum(tree_depths) / len(tree_depths)
        print(f"Loaded the following model: {model} with an average tree depth of : {average_depth}")

    if output:
        set_name = "Training"
        evaluate_model(
            model, X_train, y_train, set_name, dir_output, FILE_PERFORMANCE_TRAIN, plot=True, save=False
        )

        set_name = "Test"
        evaluate_model(
            model, X_test, y_test, set_name, dir_output, FILE_PERFORMANCE_TEST, plot=True, save=False
        )

    return model, data_train, data_test


def main():
    model_type, file_data, dir_output, n_jobs = argument_parser()
    os.makedirs(dir_output, exist_ok=True)
    train(model_type, file_data, dir_output, n_jobs)


if __name__ == "__main__":
    main()
