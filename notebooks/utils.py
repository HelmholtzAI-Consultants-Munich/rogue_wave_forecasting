############################################################
##### Imports
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
from collections import Counter

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, mean_squared_error, r2_score
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


def plot_distributions_target(target1, target2):

    nrows = 1
    ncols = 2

    plt.figure(figsize=(ncols * 4.5, nrows * 4.5))
    plt.subplots_adjust(top=0.95, hspace=0.8, wspace=0.8)
    plt.suptitle("Distribution of features")

    ax = plt.subplot(nrows, ncols, 1)
    sns.histplot(
        data=target1,
        x=target1.columns[0],
        bins=30,
        ax=ax,
        color="#3470a3",
    )

    ax = plt.subplot(nrows, ncols, 2)
    sns.histplot(
        data=target2,
        x=target2.columns[0],
        bins=30,
        ax=ax,
        color="#3470a3",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_impurity_feature_importance(importance, names, title, save=None):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by="feature_importance", ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(5, 4))

    # Plot Seaborn bar chart
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"], color="#3470a3")

    # Add chart labels
    plt.title(title)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")

    # Adjust layout to make room for labels
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


def plot_predictions(
    y_true,
    y_pred,
    title="True vs. Predicted Values",
    xlabel="True Values",
    ylabel="Predicted Values",
    textstr="",
    save=None,
):
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
        0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment="top", bbox=props
    )

    # Save the plot
    if save:
        plt.savefig(f"{save}.png", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()


def plot_correlation_matrix(data, figsize=(5, 5), annot=True, labelsize=10):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    np.fill_diagonal(mask, False)

    f, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis="both", which="major", labelsize=labelsize)
    sns.heatmap(
        round(corr, 2),
        mask=mask,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        square=True,
        ax=ax,
        annot=annot,
    )


def load_data(case, undersample, undersample_method):
    # Load and unpack the data
    with open(
        f'../data/data_case{case}{f"_{undersample_method}_undersampled" if undersample else ""}.pickle', "rb"
    ) as handle:
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


def load_data_and_model(model_type, case, undersample_method, undersample):

    # Load and unpack the data
    with open(
        f'../models/{model_type}_model_randomforest_case{case}{f"_{undersample_method}_undersampled" if undersample else ""}.pickle',
        "rb",
    ) as handle:
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

    print(f"Loaded the following model: {model}")

    if model_type == "class":
        # Predict training labels
        y_pred = model.predict(X_train)
        y_true = y_train_cat

        print(f"Balanced acc: {balanced_accuracy_score(y_true, y_pred)}")
        print(f"Macro F1 score: {f1_score(y_true, y_pred, average='macro')}")
        print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")

        # Predict test labels
        y_pred = model.predict(X_test)
        y_true = y_test_cat

        print(f"Balanced acc: {balanced_accuracy_score(y_true, y_pred)}")
        print(f"Macro F1 score: {f1_score(y_true, y_pred, average='macro')}")
        print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")

    if model_type == "reg":

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
