# Prediction and Explanation of Rogue Waves

## Project Description

**Problem Definition:** Rogue waves are extreme individual waves that can pose a threat to ships and offshore platforms. The prediction of such waves could help to avoid accidents. However, the underlying mechanisms of Rogue Wave formation are not fully understood yet.

**Project Goal:** Use AI models to predict the maximum relative wave hight 𝐻/𝐻𝑠 within the upcoming time window and use eXplainable AI methods to identify the parameters that enhance the rogue wave probability.

## Approach

To achieve the above defined project goal, we will:

- train a classification model to identify the parameters that are predictive of rogue waves 
    - use an ElasticNet model that is directly interpretable via model coefficients
    - use a Random Forest model if the linear model is not capable of modelling the data due to non-linearities in the feature-target relationshop and use Random Forest Feature Importance, SHAp and FGC for interpretation of the model results
- train a regression model for forcasting the maximum relative wave hight within the upcoming 10 min
    - depending on the classification results either use an ElasticNet or Random Forest Regressor
- perform feature selection to get a predictive model with a minimum amount of features
    - iterate via inner cross validation over all feature combinations
    - use outer cross validation for choosing the best model
    - test chosen model on test set


## Installing Requirements

```
conda create -n rogue_wave python=3.11
conda activate rogue_wave
pip install -r requirements.txt
```

## SVM Model

For the SVM model we used the ThunderSVM (https://thundersvm.readthedocs.io/en/latest/index.html) package to leverage training with GPUs.

To train the model on the cluster, we installed thunderSVM for linux:

```
git clone https://github.com/Xtra-Computing/thundersvm.git

cd thundersvm
mkdir build && cd build

cmake \ 
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 \
  ..

make -j

cd ../python
pip install .
```

Here, we set the cuda version to 12.6, because this is the latest available version on the cluster.

To be able to load the trained model locally, installed thunderSVM on mac OS:

```
git clone https://github.com/Xtra-Computing/thundersvm.git
brew install cmake libomp eigen

cd thundersvm
mkdir build && cd build

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

**NOTE:**
To make it run I had to change the following before running cmake:
1. change ```cmake_minimum_required(VERSION x.x)``` to ```cmake_minimum_required(VERSION 3.5)```
2. in the following file *thundersvm/src/thundersvm/model/svmmodel.cpp* comment the lines

```
        // sum_kernel_values_instant(coef, sv.size(), sv_start, n_sv, rho, kernel_values,instance_predict, n_classes,
        //                  batch_ins.size(),vote_device);
```