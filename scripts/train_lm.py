############################################
# imports
############################################

import os
import sys
import time

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

sys.path.append("./")
sys.path.append("../scripts/")
import utils


############################################
# SVM pipeline for GPU
############################################


def main():
    print("Setup ElasticNet...")

    n_jobs = 50
    print(f"Using {n_jobs} cores from {os.cpu_count()} available cores.")

    seed = 42
    num_cv = 5

    hyperparameter_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "max_iter": [5000],
        "tol": [1e-3, 1e-4],
        "selection": ["cyclic", "random"],
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

    print("Running ElasticNet regression with cross-validation...")

    start = time.time()
    regressor = ElasticNet(random_state=seed)
    model, cv_results = utils.run_CV(
        regressor, hyperparameter_grid, num_cv, X_train_transformed, y_train_cat, y_train, n_jobs, verbose=2
    )

    end = time.time()
    print(f"Cross-validation took {end - start:.2f} seconds")

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

    print(utils.get_model_size(model))

    print("Done.")


if __name__ == "__main__":
    main()
