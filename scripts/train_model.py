############################################
# imports
############################################

import os
import sys
import time
import pickle
import argparse

import pandas as pd

from sklearn.preprocessing import StandardScaler


sys.path.append("./")
sys.path.append("../scripts/")
import utils


############################################
# SVM pipeline for GPU
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
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "max_iter": [5000],
            "tol": [1e-3, 1e-4],
            "selection": ["cyclic", "random"],
        }
    elif model_type == "svm":
        hyperparameter_grid = {
            "kernel": ["rbf"],
            "C": [0.1, 1],
            "gamma": [0.01, 0.1, 1],
            "epsilon": [0.01, 0.1, 0.2],
        }
    elif model_type == "rf":
        hyperparameter_grid = {
            "n_estimators": [100],
            "max_depth": [10, 20, 30],
            "max_samples": [0.3, 0.4, 0.5],
            "criterion": ["friedman_mse"],
            "max_features": ["sqrt"],
            "min_samples_leaf": [1, 2, 3, 5, 10, 15],
        }
    elif model_type == "xgb":
        hyperparameter_grid = {
            "n_estimators": [100],
            "max_depth": [10, 20],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.3, 0.5],
            "colsample_bytree": [0.6, 0.8],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 1, 5],
            "reg_alpha": [0, 0.5, 1],
            "reg_lambda": [1, 2],
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


def store_predictions(model, X_train, y_train, X_test, y_test, dir_output):
    y_true_train, y_pred_train, mse_train, mae_train, r2_train, spearman_r_train = (
        utils.evaluate_best_regressor(model, X_train, y_train, dataset="Training", plot=False)
    )
    performance_train = [y_true_train, y_pred_train, mse_train, mae_train, r2_train, spearman_r_train]
    file_performance_train = f"{dir_output}/performance_train.pkl"
    with open(file_performance_train, "wb") as handle:
        pickle.dump(performance_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Training set performance: r2={r2_train}, spearman_r={spearman_r_train}")

    y_true_test, y_pred_test, mse_test, mae_test, r2_test, spearman_r_test = utils.evaluate_best_regressor(
        model, X_test, y_test, dataset="Test", plot=False
    )
    performance_test = [y_true_test, y_pred_test, mse_test, mae_test, r2_test, spearman_r_test]
    file_performance_test = f"{dir_output}/performance_test.pkl"
    with open(file_performance_test, "wb") as handle:
        pickle.dump(performance_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Test set performance: r2={r2_test}, spearman_r={spearman_r_test}")


def train(model_type, file_data, dir_output, n_jobs):

    print(f"Setup {model_type}...")
    seed = 42
    num_cv = 5

    hyperparameter_grid = get_hyperparameter_grid(model_type)

    print(f"Using {n_jobs} cores from {os.cpu_count()} available cores.")
    print(hyperparameter_grid)

    print("Loading data...")
    _, _, y_train, y_train_cat, X_train, y_test, y_test_cat, X_test = utils.load_data(file_data)

    if model_type == "lm" or model_type == "svm":
        print("Scaling data...")
        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(X_train_transformed, columns=X_train.columns)
        X_test_transformed = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test_transformed, columns=X_test.columns)

    print("Tuning hyperparameters with cross-validation...")

    start = time.time()

    regressor = get_model_instance(model_type, seed)
    model, cv_results = utils.run_CV(
        regressor, hyperparameter_grid, num_cv, X_train, y_train_cat, y_train, n_jobs, verbose=2
    )

    end = time.time()

    print(f"Model training took {end - start:.2f} seconds")

    print("Evaluating model parameter configurations...")

    cv_results.sort_values(by="score", ascending=False, inplace=True)
    cv_results.reset_index(drop=True, inplace=True)
    file_cv = f"{dir_output}/cv_results.csv"
    cv_results.to_csv(file_cv)

    print(cv_results)

    if model_type == "svm":
        store_predictions(model, X_train, y_train, X_test, y_test, dir_output)

    data_train = X_train
    data_train["AI_10min"] = y_train
    data_train["AI_10min_cat"] = y_train_cat

    data_test = X_test
    data_test["AI_10min"] = y_test
    data_test["AI_10min_cat"] = y_test_cat

    data_and_model = [data_train, data_test, model]

    file_data_model = f"{dir_output}/model_and_data.pkl"
    with open(file_data_model, "wb") as handle:
        pickle.dump(data_and_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    utils.get_model_size(model)

    print("Done.")


def main():
    model_type, file_data, dir_output, n_jobs = argument_parser()
    os.makedirs(dir_output, exist_ok=True)
    train(model_type, file_data, dir_output, n_jobs)


if __name__ == "__main__":
    main()
