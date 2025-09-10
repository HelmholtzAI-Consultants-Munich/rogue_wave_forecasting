############################################
# imports
############################################

import os
import sys
import shap
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

from imblearn.under_sampling import RandomUnderSampler

sys.path.append("./")
import utils

global _global_model, _explainer, seed
seed = 42

############################################
# SHAP pipeline
############################################


def _init_worker(model, data_background):
    _global_model = model
    _explainer = shap.TreeExplainer(
        model=_global_model,
        model_output="raw",
        feature_perturbation="tree_path_dependent",
    )

    # Use the explainer below for deep learning models
    # data_background = shap.utils.sample(data_background, 5000, random_state=seed)
    # _explainer = shap.DeepExplainer(
    #     model=_global_model,
    #     data=data_background,
    # )


def _process_batch(args):
    i, X_batch, dir_output = args
    shap_values_batch = _explainer.shap_values(X_batch)

    file_shap_batch = os.path.join(dir_output, f"shap_batch{i}.pkl")
    with open(file_shap_batch, "wb") as f:
        pickle.dump(shap_values_batch, f)
    print(f"Stored file in: {file_shap_batch}")

    return i  # Optional: used for tracking in tqdm


def run_shap(
    model,
    data_background,
    data,
    last_batch,
    batch_size,
    dir_output,
    n_jobs,
):
    # Store sampled dataset
    file_data = os.path.join(dir_output, "dataset.pkl")
    os.makedirs(os.path.dirname(file_data), exist_ok=True)
    with open(file_data, "wb") as handle:
        pickle.dump(data, handle)
    print(f"Stored file in: {file_data}")

    print("Parallel batch SHAP computation...")

    batches = []
    for i in range(0, len(data), batch_size):
        if i > last_batch:
            X_batch = data[i : i + batch_size]
            batches.append((i, X_batch, dir_output))

    with Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(
            model,
            data_background,
        ),
    ) as pool:
        list(tqdm(pool.imap(_process_batch, batches), total=len(batches)))

    # Aggregate SHAP Results
    print("Aggregate SHAP values...")
    shap_values = []

    for i in tqdm(range(0, len(data), batch_size)):
        file_shap_batch = os.path.join(dir_output, f"shap_batch{i}.pkl")
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
    file_shap = os.path.join(dir_output, "shap.pkl")
    with open(file_shap, "wb") as f:
        pickle.dump(explanation, f)

    return file_shap


def subsample(X_train, y_train, y_train_cat):
    # Combine X and y_train for resampling
    X_combined = X_train.copy()
    X_combined["target"] = y_train

    # Apply random undersampling
    rus = RandomUnderSampler(sampling_strategy="auto", random_state=seed, replacement=False)
    X_resampled, y_train_cat = rus.fit_resample(X_combined, y_train_cat)

    # Split back into features and target
    y_train = X_resampled.pop("target")
    X_train = X_resampled

    return X_train


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run SHAP.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for SHAP computation")
    parser.add_argument("--last_batch", type=int, default=-1, help="Last batch index to process")
    parser.add_argument("--dataset", type=str, default="test", help="Using 'train' or 'test' dataset")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use, default all")
    parser.add_argument("--file_data_model", type=str, help="File with model and data")
    args = parser.parse_args()

    batch_size = args.batch_size
    last_batch = args.last_batch
    dataset = args.dataset
    n_samples = args.n_samples
    file_data_model = args.file_data_model
    print(f"Run SHAP on last_batch={last_batch}, dataset={dataset}, n_samples={n_samples}")

    n_jobs = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    print(f"Detected {n_jobs} CPU cores via SLURM.")

    # Load data and model
    print("Loading data and model...")
    model, X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = utils.load_data_and_model(
        file_data_model, output=False
    )

    data_background = X_train

    if dataset == "train":
        print(f"Number of train samples {X_train.shape[0]}")
        if (n_samples is not None and n_samples > X_train.shape[0]) or n_samples is None:
            print(
                f"Warning: n_samples {n_samples} is not available or greater than available train samples {X_train.shape[0]}. Using all samples instead."
            )
            data = X_train
        else:
            data = subsample(X_train, y_train, y_train_cat)
        print(f"Using {data.shape[0]} train samples.")

    elif dataset == "test":
        print(f"Number of test samples {X_test.shape[0]}")
        if (n_samples is not None and n_samples > X_test.shape[0]) or n_samples is None:
            print(
                f"Warning: n_samples {n_samples} is greater than available test samples {X_test.shape[0]}. Using all samples instead."
            )
            data = X_test
        else:
            data = subsample(X_test, y_test, y_test_cat)
        print(f"Using {data.shape[0]} test samples.")

    dir_output = f"/lustre/groups/aiconsultants/workspace/lisa.barros/shap/"
    os.makedirs(dir_output, exist_ok=True)
    print(f"Output directory: {dir_output}")

    file_shap = run_shap(
        model=model,
        data_background=data_background,
        data=data,
        last_batch=last_batch,
        batch_size=batch_size,
        dir_output=dir_output,
        n_jobs=n_jobs,
    )
    print(f"Final SHAP values stored in {file_shap}.")


if __name__ == "__main__":
    main()
