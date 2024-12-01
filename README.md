# Code of *Practical Performative Policy Learning with Strategic Agents*



This repository provides the code necessary to reproduce the experiments presented in the paper *Practical Performative Policy Learning with Strategic Agents*. The repository is organized into two folders:

- **`synthetic/`**: Contains code for synthetic experiments.
- **`semi-synthetic/`**: Contains code for semi-synthetic experiments.

Simulation results are presented in Jupyter notebooks for experiments that are relatively less time-consuming. These notebooks include training curves and results that can be directly reviewed within the notebooks. For more computationally intensive experiments, Python scripts (`.py` files) are provided to enable seamless execution without manual intervention.



## Synthetic experiment

The **`synthetic/`** folder includes the following files:

- **`synthetic_nov_final.ipynb`**: The main notebook for reproducing experiments with the basic setting, where the manipulation cost coefficient $c\in\{0.05, 0.1, 0.15\}$.
- **`synthetic_nov_GP_final.ipynb`** & **`synthetic_nov_for_GP.ipynb`**: Contain experiments using the Gaussian process classifier and MLP as the behavior models, respectively. MLP results are included for comparison.
- **`synthetic_nov_noisy.ipynb`**: Implements experiments with noisy utility.
- **`synthetic_nov_softmax.ipynb`**: Covers experiments with softmax manipulation.
- **`coarse_discretization.py`**: Implements experiments using coarser discretization. Due to the larger sample size and number of settings compared to the basic case, this experiment is provided as a Python script.
- **`utils.py`**: Contains shared utility functions, such as model definitions, data generation, and feature manipulation.
- **`plot.ipynb`**: Generates the main figures included in the paper.





## Semi-synthetic experiment

The **`semi-synthetic/`** folder includes the following files:

- **`main.py`**: The main script for reproducing experiments with the basic setting, where the manipulation cost coefficient $c\in\{0.1, 0.15, 0.2\}$.
- **`cutoff.py`**, **`strategicGD.py`**, and **`vanillaGD.py`**: Implementations of the respective methods.
- **`utils.py`**: Contains shared utility functions, including model definitions, data generation, and feature manipulation.
- **`features_with_pred_prob.csv`**: Preprocessed dataset (excluding normalization) used in the experiments.





## Main dependency (recommended)

Ensure the following dependencies are installed to run the code:

- **PyTorch**: `torch 2.1.1+cu118`
- **NumPy**: `numpy 1.25.2`
- **Pandas**: `pandas 2.0.3`
- **scikit-learn**: `scikit-learn 1.3.0`
- **XGBoost**: `xgboost 1.7.6`
- **GPyTorch**: `gpytorch 1.13`
- **CausalML**: `causalml 0.15.1`



