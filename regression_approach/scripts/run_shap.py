############################################
# imports
############################################

import os
import sys
import shap
import pickle
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool

sys.path.append("./")
import utils

global seed
seed = 42

############################################
# SHAP pipeline
############################################


def argument_parser():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run SHAP.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for SHAP computation")
    parser.add_argument("--last_batch", type=int, default=-1, help="Last batch index to process")
    parser.add_argument("--dataset", type=str, help="Using 'train' or 'test' dataset")
    parser.add_argument(
        "--n_dataset",
        type=int,
        default=None,
        help="Number of samples to use for SHAP computation, default all",
    )
    parser.add_argument(
        "--n_background",
        type=int,
        default=None,
        help="Number of samples to use for background dataset, default all",
    )
    parser.add_argument(
        "--model_type", type=str, help="Type of model to use, i.e., 'RF', 'XGB', 'DL' or 'Kernel'"
    )
    parser.add_argument("--file_data_model", type=str, help="File with model and data")
    parser.add_argument("--dir_output", type=str, help="Directory for output files")

    args = parser.parse_args()
    batch_size = args.batch_size
    last_batch = args.last_batch
    dataset = args.dataset
    n_dataset = args.n_dataset
    n_background = args.n_background
    model_type = args.model_type
    file_data_model = args.file_data_model
    dir_output = args.dir_output

    return batch_size, last_batch, dataset, n_dataset, n_background, model_type, file_data_model, dir_output


def get_datasets(X_train, y_train, y_train_cat, X_test, y_test, y_test_cat, dataset, n_samples, n_background):
    data_background = X_train
    if (n_background is not None and n_background > X_train.shape[0]) or n_background is None:
        data_background = X_train
    else:
        data_background = subsample(X_train, y_train, y_train_cat, n_background)
        # data_background = shap.utils.sample(data_background, 5000, random_state=seed)

    if dataset == "train":
        print(f"Number of train samples {X_train.shape[0]}")
        if (n_samples is not None and n_samples > X_train.shape[0]) or n_samples is None:
            data = X_train
        else:
            data = subsample(X_train, y_train, y_train_cat, n_samples)
        print(f"Using {data.shape[0]} train samples.")

    elif dataset == "test":
        print(f"Number of test samples {X_test.shape[0]}")
        if (n_samples is not None and n_samples > X_test.shape[0]) or n_samples is None:
            data = X_test
        else:
            data = subsample(X_test, y_test, y_test_cat, n_samples)
        print(f"Using {data.shape[0]} test samples.")

    return data_background, data


def subsample(X_train, y_train, y_train_cat, n_samples):
    df = X_train.copy()
    df["target"] = y_train
    df["target_class"] = y_train_cat

    class_counts = df["target_class"].value_counts(normalize=True)
    sample_sizes = (class_counts * n_samples).round().astype(int)

    # Stratified sampling
    df = pd.concat(
        [
            df[df["target_class"] == cls].sample(n=n_samples, random_state=42)
            for cls, n_samples in sample_sizes.items()
        ]
    )

    # Drop helper columns
    data_subsampled = df.drop(columns=["target", "target_class"])

    return data_subsampled


def _init_worker(model, data_background, model_type):
    global _global_model, _explainer
    _global_model = model

    if model_type == "RF" or model_type == "XGB":
        print("Using TreeExplainer for tree-based models.")
        _explainer = shap.TreeExplainer(
            model=_global_model,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )
    elif model_type == "Kernel":
        print("Using KernelExplainer for any model.")
        print(f"Using {data_background.shape[0]} samples for background dataset.")
        _explainer = shap.KernelExplainer(
            model=_global_model.predict,
            data=data_background,
        )
    elif model_type == "DL":
        print("Using DeepExplainer for deep learning models.")
        print(f"Using {data_background.shape[0]} samples for background dataset.")
        _explainer = shap.DeepExplainer(
            model=_global_model,
            data=data_background,
        )
    else:
        raise ValueError(f"Model type {model_type} not recognized. Use 'RF', 'XGB', 'Kernel' or 'DL'.")

    print("Worker initialized.")


def _process_batch(args):
    i, X_batch, dir_output, dataset = args
    shap_values_batch = _explainer.shap_values(X_batch)

    file_shap_batch = os.path.join(dir_output, f"{dataset}_shap_batch{i}.pkl")
    with open(file_shap_batch, "wb") as f:
        pickle.dump(shap_values_batch, f)
    print(f"Stored file in: {file_shap_batch}")

    return i


def run_shap(
    model,
    data_background,
    data,
    last_batch,
    batch_size,
    dataset,
    model_type,
    dir_output,
    n_jobs,
):
    # Parallel SHAP computation
    print("Parallel batch SHAP computation...")
    batches = []
    for i in range(0, len(data), batch_size):
        if i > last_batch:
            X_batch = data[i : i + batch_size]
            batches.append((i, X_batch, dir_output, dataset))

    with Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(
            model,
            data_background,
            model_type,
        ),
    ) as pool:
        list(tqdm(pool.imap(_process_batch, batches), total=len(batches)))

    # Aggregate SHAP Results
    print("Aggregate SHAP values...")
    shap_values = []

    for i in tqdm(range(0, len(data), batch_size)):
        file_shap_batch = os.path.join(dir_output, f"{dataset}_shap_batch{i}.pkl")
        with open(file_shap_batch, "rb") as handle:
            shap_values_batch = pickle.load(handle)
        shap_values.append(shap_values_batch)
        os.remove(file_shap_batch)

    # Combine results
    shap_values = np.concatenate(shap_values, axis=0)

    # Base value (model expected value)
    expected_value = model.predict(data).mean()

    # Create SHAP Explanation
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(data), expected_value),
        data=data.values,
        feature_names=data.columns.tolist(),
    )

    # Save object to a pickle file
    file_shap = os.path.join(dir_output, f"{dataset}_shap.pkl")
    with open(file_shap, "wb") as f:
        pickle.dump(explanation, f)

    return file_shap


def main():
    batch_size, last_batch, dataset, n_dataset, n_background, model_type, file_data_model, dir_output = (
        argument_parser()
    )
    print(f"Run SHAP on last_batch={last_batch}, dataset={dataset}, model_type={model_type}")

    # n_jobs = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    n_jobs = 4
    print(f"Detected {n_jobs} CPU cores via SLURM.")

    print("Loading data and model...")
    model, X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = utils.load_data_and_model(
        file_data_model, output=False
    )

    # For XGB, set n_jobs=1 to avoid conflicts with multiprocessing
    if model_type == "XGB":
        model.set_params(n_jobs=1)

    data_background, data = get_datasets(
        X_train, y_train, y_train_cat, X_test, y_test, y_test_cat, dataset, n_dataset, n_background
    )

    os.makedirs(dir_output, exist_ok=True)
    print(f"Output directory: {dir_output}")

    file_shap = run_shap(
        model=model,
        data_background=data_background,
        data=data,
        last_batch=last_batch,
        batch_size=batch_size,
        dataset=dataset,
        model_type=model_type,
        dir_output=dir_output,
        n_jobs=n_jobs,
    )
    print(f"Final SHAP values stored in {file_shap}.")


if __name__ == "__main__":
    main()
