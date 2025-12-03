# Prediction and Explanation of Rogue Waves

## Project Overview

- **Problem.** Rogue waves are extreme Sea Surface Height events that endanger offshore operations. The physics are only partially understood, so data-driven forecasts offer a pragmatic mitigation route.
- **Objective.** Predict the maximum relative wave height `H/Hs` for the next 10 minutes and surface the oceanographic conditions that drive rogue-wave likelihood via explainable AI.

## Analytical Workflow

The `regression_approach/notebooks/` folder documents the full study. Execute the notebooks in numerical order to reproduce the pipeline:

1. **00_data_processing.ipynb** cleans raw measurements, applies physical thresholds, and exports the modelling matrix.
2. **01_linear_regression.ipynb** benchmarks ElasticNet baselines. Cross-validated `R²` peaks at ~0.06, signalling strong non-linearities.
3. **02_random_forest_regression.ipynb** tunes random forest model and computed shap values. 
4. **03_xg_boost_regression.ipynb** tunes xgboost model and computed shap values.
5. **04_svm_regression.ipynb** fits ThunderSVM regressors on GPU and computed shap values. 
6. **05_ffnn_regression.ipynb** evaluates feed-forward neural net and computed shap values.
7. **06_summary.ipynb** consolidates metrics and compares model explanations.
8. **07_stbility_and_performance.ipynb** stress-tests the chosen XGBoost model with stratified folds.


## Findings

- **Non-linearity dominates.** Tree-based and kernel models outperform linear baselines by nearly an order of magnitude in `R²`.
- **Top performers.** Random Forest, XGBoost and ThunderSVM achieve cross-validated `R²` around 0.94–0.95.
- **Stability.** Stratified resampling (5-fold, seed 42) yields stable test-target distributions and narrow error variability.

## Reproducing the Experiments

```
conda create -n rogue_wave python=3.11
conda activate rogue_wave
pip install -r requirements.txt
```

- SHAP batch jobs for the final models can be launched via `run_shap_batch_script_*.sh` after activating the environment.

## ThunderSVM Setup

We rely on [ThunderSVM](https://thundersvm.readthedocs.io/en/latest/index.html) to accelerate SVM training. Example build commands are provided for both Linux (CUDA) and macOS (CPU/OpenMP):

```
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

> **Note.** When building locally we lowered `cmake_minimum_required` to `3.5` and commented the call to `sum_kernel_values_instant` inside `thundersvm/src/thundersvm/model/svmmodel.cpp` to avoid a compilation issue with OpenMP.

## Repository Map

- `regression_approach/notebooks/` – data preparation, model training, evaluation, and stability studies.
- `regression_approach/scripts/` – utility modules, SHAP runners, and batch scripts for the cluster.
- `regression_approach/results/` – cross-validation scores and SHAP values produced by the notebooks.
