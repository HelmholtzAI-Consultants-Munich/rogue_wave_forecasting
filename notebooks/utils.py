############################################################
##### Imports
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    data = {
        "feature_names": feature_names,
        "feature_importance": feature_importance
    }
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by="feature_importance", ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(5, 4))
    
    # Plot Seaborn bar chart
    sns.barplot(
        x=fi_df["feature_importance"],
        y=fi_df["feature_names"],
        color="#3470a3"
    )
    
    # Add chart labels
    plt.title(title)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    
    # Adjust layout to make room for labels
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()



