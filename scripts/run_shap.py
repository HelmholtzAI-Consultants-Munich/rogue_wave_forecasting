############################################
# imports
############################################

import os
import sys
import time
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
    parser = argparse.ArgumentParser(description="Run SHAP.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for SHAP computation")
    parser.add_argument("--last_batch", type=int, default=-1, help="Last batch index processed for resuming")
    parser.add_argument(
        "--multi_batch",
        type=bool,
        default=True,
        help="Whether to compute SHAP values in multiple batches or a single batch",
    )
    parser.add_argument("--dataset", type=str, help="Using 'train' or 'test' dataset")
    parser.add_argument("--n_dataset", type=int, default=None, help="Number of samples for SHAP computation")
    parser.add_argument("--n_background", type=int, default=None, help="Num. samples for background dataset")
    parser.add_argument("--model_type", type=str, help="Type of input model: 'RF', 'XGB', 'DL' or 'Kernel'")
    parser.add_argument("--file_data_model", type=str, help="File with model and data")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--dir_output", type=str, help="Directory for output files")

    args = parser.parse_args()
    batch_size = args.batch_size
    last_batch = args.last_batch
    multi_batch = args.multi_batch
    dataset = args.dataset
    n_dataset = args.n_dataset
    n_background = args.n_background
    model_type = args.model_type
    file_data_model = args.file_data_model
    n_jobs = args.n_jobs
    dir_output = args.dir_output

    return (
        batch_size,
        last_batch,
        multi_batch,
        dataset,
        n_dataset,
        n_background,
        model_type,
        file_data_model,
        n_jobs,
        dir_output,
    )


def subsample(X, y, y_cat, n_samples, random_state=42):
    df = X.copy()
    df["target"] = y
    df["target_class"] = y_cat

    sample_sizes = df["target_class"].value_counts(normalize=True).mul(n_samples).round().astype(int)

    sampled_df = pd.concat(
        [
            group.sample(n=sample_sizes[cls], random_state=random_state)
            for cls, group in df.groupby("target_class")
            if cls in sample_sizes
        ]
    )

    data_subsampled_target = sampled_df["target"]
    data_subsampled = sampled_df.drop(columns=["target", "target_class"])

    return data_subsampled, data_subsampled_target


def get_datasets(X_train, y_train, y_train_cat, X_test, y_test, y_test_cat, dataset, n_samples, n_background):
    def _get_dataset(X, y, y_cat, dataset, n_samples):
        if (n_samples is not None and n_samples > X.shape[0]) or n_samples is None:
            data = X
            data_y = y
        else:
            data, data_y = subsample(X, y, y_cat, n_samples)
        print(f"Using {data.shape[0]} {dataset} samples of {X.shape[0]} in total.")
        return data, data_y

    data_background, data_background_y = _get_dataset(
        X_train, y_train, y_train_cat, "background", n_background
    )

    if dataset == "train":
        data_shap, data_shap_y = _get_dataset(X_train, y_train, y_train_cat, dataset, n_samples)

    elif dataset == "test":
        data_shap, data_shap_y = _get_dataset(X_test, y_test, y_test_cat, dataset, n_samples)

    return data_background, data_background_y, data_shap, data_shap_y


def _init_worker(model, data_background, model_type):
    global _global_model, _explainer
    _global_model = model

    if model_type == "Linear":
        print("Using LinearExplainer for linear models.")
        _explainer = shap.LinearExplainer(
            model=_global_model,
            data=data_background,
            masker=shap.maskers.Independent(data_background, max_samples=data_background.shape[0]),
            feature_perturbation="correlation_dependent",
        )
    elif model_type == "RF" or model_type == "XGB":
        print("Using TreeExplainer for tree-based models.")
        _explainer = shap.TreeExplainer(
            model=_global_model,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )
    elif model_type == "Kernel":
        print(
            f"Using KernelExplainer for any model with {data_background.shape[0]} samples for background dataset."
        )
        _explainer = shap.KernelExplainer(
            model=_global_model.predict,
            data=data_background,
        )
    elif model_type == "DL":
        print(
            f"Using DeepExplainer for deep learning models with {data_background.shape[0]} samples for background dataset."
        )
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
    data_shap,
    data_shap_y,
    multi_batch,
    last_batch,
    batch_size,
    dataset,
    model_type,
    dir_output,
    n_jobs,
):
    print("Setup SHAP computation.")
    batches = []
    for i in range(0, len(data_shap), batch_size):
        if i > last_batch and not os.path.exists(os.path.join(dir_output, f"{dataset}_shap_batch{i}.pkl")):
            print(f"Adding batch {i} to processing queue.")
            X_batch = data_shap[i : i + batch_size]
            if model_type == "DL":
                X_batch = X_batch.values if isinstance(X_batch, pd.DataFrame) else X_batch
            batches.append((i, X_batch, dir_output, dataset))
            if multi_batch == False:
                break

    if batches:
        if n_jobs == 1:
            print("Sequential batch SHAP computation...")
            _init_worker(model, data_background, model_type)
            for args in tqdm(batches, total=len(batches)):
                _process_batch(args)
        else:
            print("Parallel batch SHAP computation...")
            with Pool(
                processes=n_jobs,
                initializer=_init_worker,
                initargs=(model, data_background, model_type),
            ) as pool:
                list(tqdm(pool.imap(_process_batch, batches), total=len(batches)))

    for i in range(0, len(data_shap), batch_size):
        file_shap_batch = os.path.join(dir_output, f"{dataset}_shap_batch{i}.pkl")
        if not os.path.exists(file_shap_batch):
            raise FileNotFoundError(
                f"Expected SHAP batch file not found: {file_shap_batch}, cannot proceed with aggregation."
            )

    print("Aggregate SHAP values...")
    shap_values = []

    for i in tqdm(range(0, len(data_shap), batch_size)):
        file_shap_batch = os.path.join(dir_output, f"{dataset}_shap_batch{i}.pkl")
        with open(file_shap_batch, "rb") as handle:
            shap_values_batch = pickle.load(handle)
        if model_type == "DL":
            shap_values_batch = shap_values_batch.squeeze(-1)
        shap_values.append(shap_values_batch)

    shap_values = np.concatenate(shap_values, axis=0)

    expected_value = model.predict(data_shap).mean()

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(data_shap), expected_value),
        data=data_shap.values,
        feature_names=data_shap.columns.tolist(),
    )
    explanation.custom = {"target": data_shap_y}

    file_shap = os.path.join(dir_output, f"{dataset}_shap.pkl")
    with open(file_shap, "wb") as f:
        pickle.dump(explanation, f)

    for i in range(0, len(data_shap), batch_size):
        file_shap_batch = os.path.join(dir_output, f"{dataset}_shap_batch{i}.pkl")
        os.remove(file_shap_batch)

    return file_shap


def main():
    (
        batch_size,
        last_batch,
        multi_batch,
        dataset,
        n_dataset,
        n_background,
        model_type,
        file_data_model,
        n_jobs,
        dir_output,
    ) = argument_parser()
    print(
        f"Run SHAP on last_batch={last_batch}, multi_batch={multi_batch}, dataset={dataset}, model_type={model_type}"
    )

    n_jobs_available = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    n_jobs = min(n_jobs, n_jobs_available)
    print(f"Detected {n_jobs_available} CPU cores via SLURM. Using n_jobs={n_jobs} for SHAP computation.")

    print("Loading data and model...")
    model, X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = utils.load_data_and_model(
        file_data_model, output=False
    )

    start = time.time()
    prediction = model.predict(X_train.iloc[[0]])
    end = time.time()

    print(f"Predicting a single sample: {prediction}, which took {end - start:.6f} seconds")

    if model_type == "XGB":
        model.set_params(n_jobs=1)

    data_background, data_background_y, data_shap, data_shap_y = get_datasets(
        X_train, y_train, y_train_cat, X_test, y_test, y_test_cat, dataset, n_dataset, n_background
    )

    os.makedirs(dir_output, exist_ok=True)
    print(f"Output directory: {dir_output}")

    file_shap = run_shap(
        model=model,
        data_background=data_background,
        data_shap=data_shap,
        data_shap_y=data_shap_y,
        multi_batch=multi_batch,
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
