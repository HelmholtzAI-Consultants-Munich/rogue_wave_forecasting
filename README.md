# Prediction and Explanation of Rogue Waves

Rogue waves are exceptionally large, unexpected ocean waves, i.e. waves that exceed twice the significant wave height of the sea state around them. They rise with almost no warning, and they are a genuine hazard to shipping and offshore operations. This repository analyses a continuous radar record of roughly one million individual waves measured at
the K14 gas platform in the southern North Sea, and asks how far the sea state of the preceding half hour says anything about the largest wave in the following ten minutes.

The work has three parts:

- **Exploration.** How the sea state evolves along the record, when rogue waves cluster, how strongly consecutive waves depend on one another, and what that dependence implies for honest validation.
- **Modelling.** Several regression models — ElasticNet, SVM, random forest, XGBoost and a feed-forward neural network — trained on 17 metocean features describing waves, spectra and weather.
- **Explanation.** SHAP attributions, to see which oceanographic conditions actually drive the predictions and how stable those explanations are.

The analysis lives in `notebooks/`, meant to be run in numerical order starting with `00_data_processing.ipynb`, which also downloads the dataset. Shared plotting and modelling helpers are in `scripts/`.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or, with Homebrew, `brew install uv`.

Then clone the repository and create the environment:

```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/rogue_wave_forecasting.git
cd rogue_wave_forecasting
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`, fetches Python 3.13 if it is not already available, and creates a `.venv` with exactly the locked package versions. No separate virtual-environment step is needed.

Launch JupyterLab:

```bash
uv run jupyter lab
```

or run a script directly:

```bash
uv run python scripts/train_model.py --help
```

`uv run` uses the project environment automatically, so the `.venv` never has to be activated by hand.

### Optional components

The neural-network notebook needs TensorFlow, which is an opt-in extra:

```bash
uv sync --extra ffnn
```

The SVM model uses [ThunderSVM](https://thundersvm.readthedocs.io/), which has no wheels on PyPI and must be built from source following its own documentation. Example build commands are provided for both Linux (CUDA) and macOS (CPU/OpenMP):

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

### Notes

- `uv.lock` is resolved for Apple Silicon macOS and Linux x86_64. Other platforms need a fresh `uv lock`.
- `uv sync` removes anything not in the lock file. Use `uv sync --inexact` to keep packages you have installed alongside the project.
- `requirements.txt` lists the same direct dependencies for environments without uv (`pip install -r requirements.txt`), unpinned. `uv.lock` holds the exact versions.

## License

MIT — see [LICENSE](LICENSE).
