############################################
# imports
############################################

import os
import sys
import time

from sklearn.ensemble import RandomForestRegressor

sys.path.append("./")
sys.path.append("../scripts/")
import utils


############################################
# SVM pipeline for GPU
############################################


def main():

    print("Setup Random Forest...")

    n_jobs = 50
    print(f"Using {n_jobs} cores from {os.cpu_count()} available cores.")

    seed = 42
    num_cv = 5

    hyperparameter_grid = {
        "n_estimators": [100],
        "max_depth": [10, 20, 30],
        "max_samples": [0.3, 0.4, 0.5],
        "criterion": ["friedman_mse"],
        "max_features": ["sqrt"],
        "min_samples_leaf": [1, 2, 3, 5, 10, 15],
    }
    print(hyperparameter_grid)

    print("Loading data...")

    file_data = "../data/data_train_test.pickle"  # path to the preprocessed data
    data_train, data_test, y_train, y_train_cat, X_train, y_test, y_test_cat, X_test = utils.load_data(
        file_data
    )

    print("Running Random Forest regression with cross-validation...")

    start = time.time()
    regressor = RandomForestRegressor(random_state=seed)
    model, cv_results = utils.run_CV(
        regressor, hyperparameter_grid, num_cv, X_train, y_train_cat, y_train, n_jobs, verbose=2
    )

    end = time.time()
    print(f"Cross-validation took {end - start:.2f} seconds")

    print("Evaluating model parameter configurations...")

    cv_results.sort_values(by="score", ascending=False, inplace=True)

    print(cv_results)

    y_true_train, y_pred_train, mse_train, mae_train, r2_train, spearman_r_train = (
        utils.evaluate_best_regressor(model, X_train, y_train, dataset="Training", plot=True)
    )
    print(f"Training set performance: r2={r2_train}, spearman_r={spearman_r_train}")

    y_true_test, y_pred_test, mse_test, mae_test, r2_test, spearman_r_test = utils.evaluate_best_regressor(
        model, X_test, y_test, dataset="Test", plot=True
    )
    print(f"Test set performance: r2={r2_test}, spearman_r={spearman_r_test}")

    print(utils.get_model_size(model))

    print("Done.")


if __name__ == "__main__":
    main()
