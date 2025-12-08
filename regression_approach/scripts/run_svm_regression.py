############################################
# imports
############################################

import os
import sys
import pickle
import argparse

import pandas as pd
from thundersvm import SVR
from sklearn.preprocessing import StandardScaler


sys.path.append("./")
sys.path.append("../scripts/")
import utils


############################################
# SVM pipeline for GPU
############################################


def parse_argument_type(arg_str):
    try:
        return float(arg_str)
    except ValueError:
        return arg_str


def argument_parser():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run SVM regression.")
    parser.add_argument("--kernel", help="Kernel type for SVM")
    parser.add_argument("--C", help="Regularization parameter for SVM")
    parser.add_argument("--gamma", help="Gamma parameter for SVM")
    parser.add_argument("--epsilon", help="Epsilon parameter for SVM")
    parser.add_argument("--num_cv", type=int, default=5, help="Number of cross-validation folds")

    args = parser.parse_args()
    kernel = args.kernel
    C = args.C
    gamma = args.gamma
    epsilon = args.epsilon
    num_cv = args.num_cv

    if kernel is None:
        kernel = ["rbf"]
    else:
        kernel = [kernel]

    if C is None:
        C = [0.1, 1]
    else:
        C = [parse_argument_type(C)]

    if gamma is None:
        gamma = [0.01, 0.1, 1]
    else:
        gamma = [parse_argument_type(gamma)]

    if epsilon is None:
        epsilon = [0.01, 0.1, 0.2]
    else:
        epsilon = [parse_argument_type(epsilon)]

    return kernel, C, gamma, epsilon, num_cv


def main():

    print("Setup...")

    # We set n_jobs=1 because thundersvm uses GPU.
    n_jobs = 1
    kernel, C, gamma, epsilon, num_cv = argument_parser()

    hyperparameter_grid = {
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "epsilon": epsilon,
    }

    print(hyperparameter_grid)

    print("Loading data...")

    file_data = "../data/data_train_test.pickle"  # path to the preprocessed data
    data_train, data_test, y_train, y_train_cat, X_train, y_test, y_test_cat, X_test = utils.load_data(
        file_data
    )

    print("Scaling data...")

    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    print("Running SVM regression with cross-validation...")

    regressor = SVR()
    model, cv_results = utils.run_CV(
        regressor, hyperparameter_grid, num_cv, X_train_transformed, y_train_cat, y_train, n_jobs, verbose=2
    )

    print("Evaluating model parameter configurations...")

    cv_results.sort_values(by="score", ascending=False, inplace=True)

    print(cv_results)

    y_true_train, y_pred_train, mse_train, mae_train, r2_train, spearman_r_train = (
        utils.evaluate_best_regressor(model, X_train_transformed, y_train, dataset="Training", plot=True)
    )
    print(f"Training set performance: r2={r2_train}, spearman_r={spearman_r_train}")

    y_true_test, y_pred_test, mse_test, mae_test, r2_test, spearman_r_test = utils.evaluate_best_regressor(
        model, X_test_transformed, y_test, dataset="Test", plot=True
    )
    print(f"Test set performance: r2={r2_test}, spearman_r={spearman_r_test}")

    print("Saving results...")

    file_cv = f"../results/svm/cv_results_{gamma}_{epsilon}.csv"
    cv_results.to_csv(file_cv)

    data_train = pd.DataFrame(X_train_transformed, columns=X_train.columns)
    data_train["AI_10min"] = y_train
    data_train["AI_10min_cat"] = y_train_cat

    data_test = pd.DataFrame(X_test_transformed, columns=X_test.columns)
    data_test["AI_10min"] = y_test
    data_test["AI_10min_cat"] = y_test_cat

    data_and_model = [data_train, data_test, model]

    file_data_model = f"../results/svm/model_and_data_{gamma}_{epsilon}.pickle"
    with open(file_data_model, "wb") as handle:
        pickle.dump(data_and_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(utils.get_model_size(model))

    performance_train = [y_true_train, y_pred_train, mse_train, mae_train, r2_train, spearman_r_train]
    file_performance_train = f"../results/svm/performance_train_{gamma}_{epsilon}.pickle"
    with open(file_performance_train, "wb") as handle:
        pickle.dump(performance_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    performance_test = [y_true_test, y_pred_test, mse_test, mae_test, r2_test, spearman_r_test]
    file_performance_test = f"../results/svm/performance_test_{gamma}_{epsilon}.pickle"
    with open(file_performance_test, "wb") as handle:
        pickle.dump(performance_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")


if __name__ == "__main__":
    main()
