############################################
# imports
############################################

import os
import sys
import shap
import pickle

import numpy as np

from tqdm import tqdm
from multiprocessing import Pool

sys.path.append("./")
import utils

############################################
# SHAP pipeline
############################################


def _init_worker(model):
    global _global_model, _explainer
    _global_model = model
    _explainer = shap.TreeExplainer(
        model=_global_model, model_output="raw", feature_perturbation="tree_path_dependent"
    )


def _process_batch(args):
    i, X_batch, model_type, dir_output = args
    shap_values_batch = _explainer.shap_values(X_batch)

    file_shap_batch = os.path.join(dir_output, f"shap_batch{i}.pkl")
    with open(file_shap_batch, "wb") as f:
        pickle.dump(shap_values_batch, f)
    print(f"Stored file in: {file_shap_batch}")

    return i  # Optional: used for tracking in tqdm


def run_shap(
    model_type,
    case,
    undersample_method,
    undersample,
    last_batch,
    batch_size,
    dir_output,
    n_jobs,
    seed,
):
    # Load data and model
    print("Loading data and model...")
    model_full, X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = utils.load_data_and_model(
        model_type, case, undersample_method, undersample, output=False
    )
    print(f"Number of samples {X_train.shape[0]}")

    # Store sampled dataset
    file_data = os.path.join(dir_output, "dataset.pkl")
    os.makedirs(os.path.dirname(file_data), exist_ok=True)
    with open(file_data, "wb") as handle:
        pickle.dump(X_train, handle)
    print(f"Stored file in: {file_data}")

    print("Parallel batch SHAP computation...")

    batches = []
    for i in range(0, len(X_train), batch_size):
        if i > last_batch:
            X_batch = X_train[i : i + batch_size]
            batches.append((i, X_batch, model_type, dir_output))

    with Pool(processes=n_jobs, initializer=_init_worker, initargs=(model_full,)) as pool:
        list(tqdm(pool.imap(_process_batch, batches), total=len(batches)))

    # Aggregate SHAP Results
    print("Aggregate SHAP values...")
    shap_values = []

    for i in tqdm(range(0, len(X_train), batch_size)):
        file_shap_batch = os.path.join(dir_output, f"shap_batch{i}.pkl")
        with open(file_shap_batch, "rb") as handle:
            shap_values_batch = pickle.load(handle)
        shap_values.append(shap_values_batch)
        # os.remove(file_shap_batch)

    # Combine results
    shap_values = np.concatenate(shap_values, axis=0)

    # Save object to a pickle file
    file_shap = os.path.join(dir_output, "shap.pkl")
    with open(file_shap, "wb") as f:
        pickle.dump(shap_values, f)

    return file_shap


def main():
    seed = 42
    n_jobs = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    print(f"Detected {n_jobs} CPU cores via SLURM.")

    model_type = "reg"
    undersample = False
    undersample_method = None
    case = 3

    last_batch = -1  # Change this if you want to resume from a specific index
    batch_size = 100

    dir_output = f"/lustre/groups/aiconsultants/workspace/lisa.barros/{model_type}/shap/"
    os.makedirs(dir_output, exist_ok=True)
    print(f"Output directory: {dir_output}")

    file_shap = run_shap(
        model_type=model_type,
        case=case,
        undersample_method=undersample_method,
        undersample=undersample,
        last_batch=last_batch,
        batch_size=batch_size,
        dir_output=dir_output,
        n_jobs=n_jobs,
        seed=seed,
    )
    print(f"Final SHAP values stored in {file_shap}.")


if __name__ == "__main__":
    main()
