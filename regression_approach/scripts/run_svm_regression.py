############################################
# imports
############################################

import os
import sys
import pickle
import shap

import pandas as pd
from thundersvm import SVR
from sklearn.preprocessing import StandardScaler


sys.path.append("./")
sys.path.append("../scripts/")
import utils


############################################
# SVM pipeline for GPU
############################################

print("Setup...")

n_jobs = 4
print(
    f"Using {n_jobs} cores from {os.cpu_count()} available cores."
)  # how many CPU cores are available on the current machine

seed = 42
num_cv = 5

hyperparameter_grid = {
    "kernel": ["rbf", "poly"],  # RBF is flexible for non-linear patterns
    "C": [0.1, 1, 10],  # Regularization strength (low = more regularization)
    "gamma": ["auto", 0.01, 0.1, 1],  # Kernel coefficient for 'rbf', 'poly'
    "epsilon": [0.01, 0.1, 0.2],  # Margin of tolerance where no penalty is given
}

print("Loading data...")

file_data = "../data/data_train_test.pickle"  # path to the preprocessed data
data_train, data_test, y_train, y_train_cat, X_train, y_test, y_test_cat, X_test = utils.load_data(file_data)

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

file_cv = f"../results/svm/cv_results.csv"

cv_results.sort_values(by="score", ascending=False)
cv_results.to_csv(file_cv)

print(cv_results)

print("Saving results...")

data_train = pd.DataFrame(X_train_transformed, columns=X_train.columns)
data_train["AI_10min"] = y_train
data_train["AI_10min_cat"] = y_train_cat

data_test = pd.DataFrame(X_test_transformed, columns=X_test.columns)
data_test["AI_10min"] = y_test
data_test["AI_10min_cat"] = y_test_cat

data_and_model = [data_train, data_test, model]

file_data_model = f"../results/svm/model_and_data.pickle"
with open(file_data_model, "wb") as handle:
    pickle.dump(data_and_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(utils.get_model_size(model))

print("Done.")
