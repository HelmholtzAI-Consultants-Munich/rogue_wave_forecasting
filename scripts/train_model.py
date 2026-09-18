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

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

DIR_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(DIR_ROOT / "scripts"))

print(f"DIR_ROOT: {DIR_ROOT}")

import utils

from constants import (
    COLUMN_TARGET,
    COLUMN_FEATURES,
    COLUMN_FOLD,
    PURGE_GAP,
    NUM_CV,
    DIR_RESULTS,
    FILE_DATA_PROCESSED,
    FILE_CV_RESULTS,
    FILE_MODEL_AND_DATA,
    FILE_PERFORMANCE_TEST_CSV,
    FILE_PERFORMANCE_TRAIN_CSV,
    HYPERPARAMETER_GRID_LM,
    HYPERPARAMETER_GRID_SVM,
    HYPERPARAMETER_GRID_RF,
    HYPERPARAMETER_GRID_XGB,
)

SEED = 42

############################################
# Train Model
############################################


def argument_parser():
    parser = argparse.ArgumentParser(description="Run SHAP.")
    parser.add_argument("--model_type", type=str, help="Type of model to train")
    parser.add_argument("--cv_type", type=str, help="Type of cross-validation")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")

    args = parser.parse_args()
    model_type = args.model_type
    cv_type = args.cv_type
    file_data = FILE_DATA_PROCESSED
    dir_output = f"{DIR_RESULTS}/{model_type}_{cv_type}"
    n_jobs = args.n_jobs

    return model_type, cv_type, file_data, dir_output, n_jobs


def get_hyperparameter_grid(model_type):
    if model_type == "lm":
        hyperparameter_grid = HYPERPARAMETER_GRID_LM

    elif model_type == "svm":
        hyperparameter_grid = HYPERPARAMETER_GRID_SVM

    elif model_type == "rf":
        hyperparameter_grid = HYPERPARAMETER_GRID_RF

    elif model_type == "xgb":
        hyperparameter_grid = HYPERPARAMETER_GRID_XGB
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


def load_data(file_data, column_target, column_features, column_fold):

    with open(file_data, "rb") as handle:
        data_train, data_test = pickle.load(handle)

    X_train, y_train = data_train[column_features], data_train[column_target]
    X_test, y_test = data_test[column_features], data_test[column_target]
    cv_groups = data_train[column_fold].to_numpy()

    return data_train, data_test, X_train, y_train, X_test, y_test, cv_groups


def run_CV(model, hyperparameter_grid, cv_type, num_cv, X, y, groups, n_jobs, verbose=0):

    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

    if cv_type == "grouped":

        mask = groups.between(1, num_cv)
        X = X.loc[mask]
        y = y.loc[mask]
        groups = groups.loc[mask]

        skf = list(GroupKFold(n_splits=num_cv).split(X, y, groups=groups))
    elif cv_type == "time_series":
        skf = list(
            TimeSeriesSplit(
                n_splits=num_cv, test_size=len(X) // 10, gap=PURGE_GAP, max_train_size=None
            ).split(X, y)
        )

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

    # Get all CV results
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


def evaluate_model(model, X, y, set_name, dir_output, filename_plot, filename_csv, plot=True, save=False):
    print(f"Evaluate on {set_name} Set")

    y_pred = model.predict(X)

    mse = round(mean_squared_error(y, y_pred), 3)
    mae = round(mean_absolute_error(y, y_pred), 3)
    r2 = round(r2_score(y, y_pred), 3)
    spearman_r = round(spearmanr(y, y_pred).correlation, 3)

    print(
        f"{set_name} set performance: r2={r2:.3f}, spearman_r={spearman_r:.3f}, mse={mse:.3f}, mae={mae:.3f}"
    )

    if plot:

        textstr = f"$MSE={mse}$\n$MAE={mae}$\n$R^2={r2}$\n$Spearman\\ R={spearman_r}$"
        fig, ax = utils.plot_predictions(y_true=y, y_pred=y_pred, textstr=textstr)

    if save:
        output = [y, y_pred, mse, mae, r2, spearman_r]
        with open(f"{dir_output}/{filename_csv}", "wb") as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fig.savefig(f"{dir_output}/{filename_plot}", bbox_inches="tight", dpi=300)


def get_model_size(model):
    serialized = pickle.dumps(
        model,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    size_gib = len(serialized) / 1024**3
    print(f"Serialized model size: {size_gib:.4f} GiB")

    return size_gib


def train(
    model_type,
    column_target,
    column_features,
    column_fold,
    cv_type,
    num_cv,
    file_data,
    file_model_and_data,
    file_cv_results,
    file_performance_train_csv,
    file_performance_test_csv,
    dir_output,
    n_jobs,
):

    print(f"Setup {model_type}...")
    hyperparameter_grid = get_hyperparameter_grid(model_type)

    print(f"Storing results in {dir_output} and using {n_jobs} cores from {os.cpu_count()} available cores.")
    print(hyperparameter_grid)

    print(f"Loading data from {file_data}...")
    data_train, data_test, X_train, y_train, X_test, y_test, cv_groups = load_data(
        file_data, column_target, column_features, column_fold
    )

    print(
        f"{len(X_train):,} train rows in folds {sorted(set(cv_groups))}, "
        f"{len(X_test):,} test rows, {X_train.shape[1]} features."
    )

    print("Getting model instance...")
    base_model = get_model_instance(model_type, SEED)
    needs_scaling = model_type in {"lm", "svm"}
    pipeline = Pipeline(
        [("scale", StandardScaler() if needs_scaling else "passthrough"), ("model", base_model)]
    )
    hyperparameter_grid = {f"model__{name}": values for name, values in hyperparameter_grid.items()}

    print("Tuning hyperparameters with cross-validation...")
    start = time.time()
    model, cv_results = run_CV(
        pipeline, hyperparameter_grid, cv_type, num_cv, X_train, y_train, cv_groups, n_jobs, verbose=2
    )
    end = time.time()
    print(f"Model training took {end - start:.2f} seconds")

    print("Evaluating model parameter configurations...")
    cv_results = cv_results.sort_values("rmse", ascending=False).reset_index(drop=True)
    cv_results.to_csv(f"{dir_output}/{file_cv_results}", index=False)
    print(cv_results)

    if model_type == "svm":
        evaluate_model(
            model=model,
            X=X_train,
            y=y_train,
            set_name="Training",
            dir_output=dir_output,
            filename_csv=file_performance_train_csv,
            filename_plot=None,
            plot=False,
            save=True,
        )
        evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            set_name="Test",
            dir_output=dir_output,
            filename_csv=file_performance_test_csv,
            filename_plot=None,
            plot=False,
            save=True,
        )

    with open(f"{dir_output}/{file_model_and_data}", "wb") as handle:
        pickle.dump([data_train, data_test, model], handle, protocol=pickle.HIGHEST_PROTOCOL)

    size_gib = get_model_size(model)

    print("Done.")


def load_data_and_model(
    file_data_model,
    column_target,
    column_features,
    file_performance_train_plot=None,
    file_performance_train_csv=None,
    file_performance_test_plot=None,
    file_performance_test_csv=None,
    dir_output=None,
    output=True,
):

    # Load and unpack the data
    with open(file_data_model, "rb") as handle:
        data_train, data_test, model = pickle.load(handle)

    X_train, y_train = data_train[column_features], data_train[column_target]
    X_test, y_test = data_test[column_features], data_test[column_target]

    if isinstance(model, RandomForestRegressor):
        tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
        average_depth = sum(tree_depths) / len(tree_depths)
        print(f"Loaded the following model: {model} with an average tree depth of : {average_depth}")

    if output:
        evaluate_model(
            model=model,
            X=X_train,
            y=y_train,
            set_name="Training",
            dir_output=dir_output,
            filename_plot=file_performance_train_plot,
            filename_csv=file_performance_train_csv,
            plot=True,
            save=False,
        )
        evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            set_name="Test",
            dir_output=dir_output,
            filename_plot=file_performance_test_plot,
            filename_csv=file_performance_test_csv,
            plot=True,
            save=False,
        )

    return model, data_train, data_test, X_train, y_train, X_test, y_test


def main():
    model_type, cv_type, n_jobs = argument_parser()
    os.makedirs(DIR_RESULTS, exist_ok=True)
    train(
        model_type=model_type,
        column_target=COLUMN_TARGET,
        column_features=COLUMN_FEATURES,
        column_fold=COLUMN_FOLD,
        cv_type=cv_type,
        num_cv=NUM_CV,
        file_data=FILE_DATA_PROCESSED,
        file_model_and_data=FILE_MODEL_AND_DATA,
        file_cv_results=FILE_CV_RESULTS,
        file_performance_train_csv=FILE_PERFORMANCE_TRAIN_CSV,
        file_performance_test_csv=FILE_PERFORMANCE_TEST_CSV,
        dir_output=DIR_RESULTS,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    main()
