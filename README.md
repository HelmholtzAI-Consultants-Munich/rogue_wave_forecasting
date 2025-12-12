# Prediction and Explanation of Rogue Waves

## Project Overview

- **Problem.** Rogue waves are extreme Sea Surface Height events that endanger offshore operations. The physics are only partially understood, so data-driven forecasts offer a pragmatic mitigation route.
- **Objective.** Predict the maximum relative wave height `H/Hs` for the next 10 minutes and explore the oceanographic conditions that drive rogue-wave formation via explainable AI.

## Repository Structure

### Notebooks

Data preparation, model training, evaluation, and stability studies.

1. **00_data_processing.ipynb** cleans data, selects features, and exports the modelling matrix.
2. **01_linear_regression.ipynb** train, evaluate and interpret ELasticNet model with SHAP
3. **02_random_forest_regression.ipynb** train, evaluate and interpret Random Forest model with SHAP
4. **03_xg_boost_regression.ipynb** train, evaluate and interpret XGBoost model with SHAP
5. **04_svm_regression.ipynb** train, evaluate and interpret SVM model with SHAP
6. **05_ffnn_regression.ipynb** train, evaluate and interpret feed-foreward neural network model with SHAP
7. **06_summary.ipynb** consolidates metrics and compares model explanations.
8. **07_stbility_and_performance.ipynb** stress-tests the chosen XGBoost model with stratified folds.

### Scripts

Utility modules, SHAP runners, and batch scripts for the cluster.

### Results

Cross-validation scores and SHAP values produced by the notebooks.
Due to size of the files, the content of this folder is provided along the release.


## Reproducing the Experiments

```
conda create -n rogue_wave python=3.11
conda activate rogue_wave
pip install -r requirements.txt
```

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

> **Note.** When building locally on MacOs we lowered `cmake_minimum_required` to `3.5` and commented the call to `sum_kernel_values_instant` inside `thundersvm/src/thundersvm/model/svmmodel.cpp` to avoid a compilation issue with OpenMP.

