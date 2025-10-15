############################################################
##### Imports
############################################################

import os
import math
import pickle
import shap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

############################################################
##### Utility Functions
############################################################


def plot_distributions(dataset, ncols):

    nrows = int(np.ceil(len(dataset.columns) / ncols))

    plt.figure(figsize=(ncols * 4.5, nrows * 4.5))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle("Distribution of features")

    for n, feature in enumerate(dataset.columns):
        # add a new subplot iteratively
        ax = plt.subplot(nrows, ncols, n + 1)
        if dataset[feature].nunique() < 5 or isinstance(dataset[feature].dtype, pd.CategoricalDtype):
            sns.countplot(
                data=dataset,
                x=feature,
                hue=feature,
                palette="Blues_r",
                ax=ax,
            )
            # ax.legend(bbox_to_anchor=(1, 1), loc=2)
        else:
            sns.histplot(
                data=dataset,
                x=feature,
                bins=30,
                ax=ax,
                color="#3470a3",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def load_data(file_data):
    print("Loading data...")

    # Load and unpack the data
    with open(file_data, "rb") as handle:
        data = pickle.load(handle)

    data_train, data_test = data
    y_train = data_train["AI_10min"]
    y_train_cat = data_train["AI_10min_cat"]
    y_test = data_test["AI_10min"]
    y_test_cat = data_test["AI_10min_cat"]

    X_train = data_train.drop(columns=["AI_10min", "AI_10min_cat"])
    X_test = data_test.drop(columns=["AI_10min", "AI_10min_cat"])

    print("\nTraining dataset target distribution:")
    print(Counter(data_train["AI_10min_cat"]))

    print("\nTest dataset target distribution:")
    print(Counter(data_test["AI_10min_cat"]))

    return data_train, data_test, y_train, y_train_cat, X_train, y_test, y_test_cat, X_test


def run_CV(model, hyperparameter_grid, num_cv, X, y_train_cat, y_train, n_jobs, verbose=0):
    # Tune hyperparameters
    skf = StratifiedKFold(n_splits=num_cv).split(X, y_train_cat)

    gridsearch_classifier = GridSearchCV(model, hyperparameter_grid, cv=skf, n_jobs=n_jobs, verbose=verbose)

    if isinstance(model, ClassifierMixin):
        gridsearch_classifier.fit(X, y_train_cat)
    elif isinstance(model, RegressorMixin):
        gridsearch_classifier.fit(X, y_train)

    # Take the best estimator
    model = gridsearch_classifier.best_estimator_

    # Collect CV Results
    cv_results = pd.concat(
        [
            pd.DataFrame(gridsearch_classifier.cv_results_["params"]),
            pd.DataFrame(gridsearch_classifier.cv_results_["mean_test_score"], columns=["score"]),
        ],
        axis=1,
    )

    return model, cv_results


def evaluate_best_regressor(model, X, y, dataset, plot=True):
    # Predict labels
    y_pred = model.predict(X)
    y_true = y

    mse = round(mean_squared_error(y_true, y_pred), 3)
    mae = round(mean_absolute_error(y_true, y_pred), 3)
    r2 = round(r2_score(y_true, y_pred), 3)
    spearman_r = round(spearmanr(y_true, y_pred).correlation, 3)

    print(f"Evaluate on {dataset} Set")

    if plot:
        textstr = f"$MSE={mse}$\n$MAE={mae}$\n$R^2={r2}$\n$Spearman\\ R={spearman_r}$"
        plot_predictions(y_true=y_true, y_pred=y_pred, textstr=textstr)

    return y_true, y_pred, mse, mae, r2, spearman_r


def get_model_size(model):
    file = f"./model_size_test.pickle"
    with open(file, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Get the size of the saved file
    model_size_gb = os.path.getsize(file) / (1024**3)
    print(f"Model size on disk: {model_size_gb:.4f} GB")

    os.remove(file)


def plot_predictions(
    y_true,
    y_pred,
    title="True vs. Predicted Values",
    xlabel="True Values",
    ylabel="Predicted Values",
    textstr="",
    save=None,
):
    # Convert to 1D arrays for DL models
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Create a figure with specified size
    plt.figure(figsize=(4, 4))

    # Create a scatter plot with Seaborn
    sns.set_theme(style="white")
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, color="#6693F5")

    # Add a reference line y = x
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

    # Set plot labels and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    plt.text(
        0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=8, verticalalignment="top", bbox=props
    )

    # Save the plot
    if save:
        plt.savefig(f"{save}.png", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()


def load_data_and_model(file_data_model, output=True):

    # Load and unpack the data
    with open(file_data_model, "rb") as handle:
        data_and_model = pickle.load(handle)

    data_train, data_test, model = data_and_model

    y_train = data_train["AI_10min"]
    y_train_cat = data_train["AI_10min_cat"]
    y_test = data_test["AI_10min"]
    y_test_cat = data_test["AI_10min_cat"]

    X_train = data_train.drop(columns=["AI_10min", "AI_10min_cat"])
    X_test = data_test.drop(columns=["AI_10min", "AI_10min_cat"])

    print("Training dataset target distribution:")
    print(Counter(y_train_cat))

    print("Test dataset target distribution:")
    print(Counter(y_test_cat))

    if isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        tree_depths = [estimator.tree_.max_depth for estimator in model.estimators_]
        average_depth = sum(tree_depths) / len(tree_depths)
        print(f"Loaded the following model: {model} with an average tree depth of : {average_depth}")

    if output:
        # Predict labels
        y_pred = model.predict(X_train)
        y_true = y_train

        print(f"Train set MSE: {round(mean_squared_error(y_true, y_pred), 3)}")
        print(f"Train set R^2: {round(r2_score(y_true, y_pred), 3)}")
        print(f"Train set Spearman R: {round(spearmanr(y_true, y_pred).correlation, 3)}")

        plot_predictions(y_true=y_true, y_pred=y_pred, textstr=f"$R^2={round(r2_score(y_true, y_pred), 3)}$")

        # Predict labels
        y_pred = model.predict(X_test)
        y_true = y_test

        print(f"Test set MSE: {round(mean_squared_error(y_true, y_pred), 3)}")
        print(f"Test set R^2: {round(r2_score(y_true, y_pred), 3)}")
        print(f"Test set Spearman R: {round(spearmanr(y_true, y_pred).correlation, 3)}")

        plot_predictions(y_true=y_true, y_pred=y_pred, textstr=f"$R^2={round(r2_score(y_true, y_pred), 3)}$")

    return model, X_train, y_train, y_train_cat, X_test, y_test, y_test_cat


def plot_shap_dependence(explanation):
    # 1. Compute mean absolute SHAP values
    mean_shap = np.abs(explanation.values).mean(axis=0)

    # 2. Sort features by descending mean SHAP
    sorted_indices = np.argsort(-mean_shap)
    sorted_features = [explanation.feature_names[i] for i in sorted_indices]

    # 3. Set up grid layout
    n_features = len(sorted_features)
    cols = 6
    rows = math.ceil(n_features / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.flatten()

    # 4. Generate SHAP dependence plots in the grid
    for i, feature in enumerate(sorted_features):
        shap.dependence_plot(
            ind=feature,
            shap_values=explanation.values,
            features=explanation.data,
            feature_names=explanation.feature_names,
            ax=axes[i],
            show=False,
        )
        axes[i].set_title(f"Dependence: {feature}", fontsize=10)

    # 5. Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
