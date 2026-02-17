# Prediction and Explanation of Rogue Waves

## Project Overview

- **Problem.** Rogue waves are extreme Sea Surface Height events that endanger offshore operations. The physics are only partially understood, so data-driven forecasts offer a pragmatic mitigation route.
- **Objective.** Predict the maximum relative wave height `H/Hs` for the next 10 minutes and explore the oceanographic conditions that drive rogue-wave formation via explainable AI.

## Repository Structure

### Data

- **00_data_processing.ipynb** downloads the raw dataset from the [rogue_wave_prediction release (v1.0.2)](https://github.com/HelmholtzAI-Consultants-Munich/rogue_wave_prediction/releases/download/v1.0.2/data.zip), processes it, and writes **`data/data_train_test.pickle`**. All other notebooks and training scripts expect this file (or the same train/test split format). Run this notebook first from the `notebooks/` directory.

### Notebooks

Data preparation, model training, evaluation, and stability studies.

1. **00_data_processing.ipynb** cleans data, selects features, and exports the modelling matrix.
2. **01_linear_regression.ipynb** — Train, evaluate and interpret ElasticNet model with SHAP.
3. **02_svm_regression.ipynb** — Train, evaluate and interpret SVM model with SHAP.
4. **03_random_forest_regression.ipynb** — Train, evaluate and interpret Random Forest model with SHAP.
5. **04_xg_boost_regression.ipynb** — Train, evaluate and interpret XGBoost model with SHAP.
6. **05_ffnn_regression.ipynb** — Train, evaluate and interpret feed-forward neural network model with SHAP.
7. **06_summary.ipynb** — Consolidates metrics and compares model explanations.
8. **07_performance_stability.ipynb** — Stress-tests the chosen XGBoost model with stratified folds.


### Scripts

Python and shell scripts in `scripts/` support both local runs and cluster (SLURM) jobs:

- **`utils.py`** — Shared helpers (e.g. data loading).
- **`train_model.py`** — Generic training entry point (where applicable).
- **`train_ffnn.py`**, **`train_ffnn.sh`** — FFNN training (TensorFlow/Keras, scikeras, GridSearchCV); the `.sh` script is set up for SLURM (e.g. DKRZ Levante).
- **`train_lm.sh`**, **`train_rf.sh`**, **`train_svm.sh`**, **`train_xgb.sh`** — SLURM batch scripts for ElasticNet, Random Forest, SVM, and XGBoost.
- **`run_shap.py`** — SHAP computation with options for batch processing: `--batch_number`, `--multi_batch`, `--batch_multiprocessing` (see script help for cluster-friendly usage).
- **`run_shap_ffnn.sh`**, **`run_shap_lm.sh`**, **`run_shap_rf.sh`**, **`run_shap_svm.sh`**, **`run_shap_xgb.sh`** — Example SLURM scripts that call `run_shap.py` (e.g. job arrays for SVM SHAP batches).

Paths, accounts, and partition names in the `.sh` files are examples; adjust them for your cluster and workspace.

### Results

Cross-validation scores and SHAP values produced by the notebooks.
Due to size of the files, the content of this folder is provided along the release.


## Reproducing the Experiments

```
conda create -n rogue_wave python=3.11
conda activate rogue_wave
pip install -r requirements.txt
```

For the **FFNN model** and notebook **05_ffnn_regression.ipynb**, install TensorFlow and scikeras as well:

```
pip install tensorflow scikeras
```

**Workflow:** Run **00_data_processing.ipynb** first (from `notebooks/`) to produce `data/data_train_test.pickle`. Then run the regression notebooks in order, or use the training scripts in `scripts/` for batch/cluster runs.

## ThunderSVM Setup

We rely on [ThunderSVM](https://thundersvm.readthedocs.io/en/latest/index.html) to accelerate SVM training. Example build commands are provided for both Linux (CUDA) and macOS (CPU/OpenMP):

```
conda activate rogue_wave

git clone https://github.com/Xtra-Computing/thundersvm.git

cd thundersvm
mkdir build && cd build

# Linux + CUDA 12.6
cmake \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  ..
make -j

# macOS (CPU-only)
cmake \
  -DUSE_CUDA=OFF \
  -DEigen3_DIR=/opt/homebrew/share/eigen3/cmake \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  ..
make -j

cd ../python
pip install .
```

> **Note.** When building locally on macOS we lowered `cmake_minimum_required` to `3.5` and commented the call to `sum_kernel_values_instant` inside `thundersvm/src/thundersvm/model/svmmodel.cpp` to avoid a compilation issue with OpenMP.

## Running on a cluster (SLURM)

The shell scripts in `scripts/` use SLURM directives (`#SBATCH`). Submit from the `scripts/` directory after setting paths and `#SBATCH --account` (and optionally partition/qos) to match your cluster. Training scripts (`train_*.sh`) run a single job; SHAP scripts such as `run_shap_svm.sh` use job arrays (e.g. one task per batch). Pre-trained models and `data_train_test.pickle` must be available at the paths used inside each script.

