############################################
# imports
############################################

import os
import sys
import argparse
import pickle

## Import the Forest-Guided Clustering package
from fgclustering import (
    forest_guided_clustering,
    DistanceRandomForestProximity,
    ClusteringClara,
)

sys.path.append("./")
import utils

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

############################################
# FGC pipeline
############################################


def run_FGC(model_type, case, undersample_method, undersample, k, dir_output, n_jobs, seed):
    # Load data and model
    print("Loading data and model...")
    model_full, X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = utils.load_data_and_model(
        model_type, case, undersample_method, undersample, output=False
    )

    # Store sampled dataset
    file_data = os.path.join(dir_output, f"dataset_k{k}.pkl")
    with open(file_data, "wb") as handle:
        pickle.dump([X_train, y_train], handle)

    print(f"Stored file in: {file_data}")

    fgc = forest_guided_clustering(
        k=k,
        estimator=model_full,
        X=X_train,
        y=y_train,
        clustering_distance_metric=DistanceRandomForestProximity(),
        clustering_strategy=ClusteringClara(
            sub_sample_size=0.1,
            sampling_iter=5,
            sampling_target=y_train_cat,
            method="fasterpam",
            init="random",
            max_iter=100,
        ),
        JI_bootstrap_iter=50,
        JI_bootstrap_sample_size=0.8,
        JI_discart_value=0.6,
        n_jobs=n_jobs,
        random_state=seed,
    )

    # Save object to a pickle file
    file_fgc = os.path.join(dir_output, f"fgc_k{k}.pkl")
    with open(file_fgc, "wb") as handle:
        pickle.dump(fgc, handle)

    return file_fgc


def main():
    seed = 42

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Forest-Guided Clustering with k.")
    parser.add_argument("--k", type=int, default=2, help="Number of clusters to use (default: 2)")
    args = parser.parse_args()

    k = args.k
    print(f"Run FGC on k={k}")

    n_cores = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    n_jobs = 1
    print(f"Detected {n_cores} CPU cores via SLURM and using {n_jobs} cores.")

    model_type = "reg"
    undersample = False
    undersample_method = None
    case = 3

    dir_output = f"/lustre/groups/aiconsultants/workspace/lisa.barros/{model_type}/fgc/"
    os.makedirs(dir_output, exist_ok=True)
    print(f"Output directory: {dir_output}")

    file_fgc = run_FGC(
        model_type=model_type,
        case=case,
        undersample_method=undersample_method,
        undersample=undersample,
        k=k,
        dir_output=dir_output,
        n_jobs=n_jobs,
        seed=seed,
    )

    print(f"Final FGC values stored in {file_fgc}.")


if __name__ == "__main__":
    main()

