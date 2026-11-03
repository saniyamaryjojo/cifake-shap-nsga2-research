# CNN Accuracy–SHAP Stability Optimization Using NSGA-II

## Project Overview

This project investigates the trade-off between neural network classification accuracy and SHAP explanation stability using multiobjective evolutionary optimization.

## Dataset

- CIFAKE image dataset from Hugging Face
- Binary classification: AI-generated vs. real images
- 2,000 training images and 200 validation images used in the initial optimization experiment

## Methodology

A small convolutional neural network is trained using PyTorch.

NSGA-II, implemented using pymoo, optimizes two hyperparameters:
- Learning rate
- Weight decay

The objectives are to minimize validation classification error and SHAP explanation instability.

SHAP stability is measured by comparing attribution rankings between original and slightly perturbed images using Spearman rank correlation.

## Initial Experiment

- Population size: 6
- Generations: 3
- Candidate evaluations: 18
- Training: 1 epoch per candidate
- Computing platform: Nibi cluster, using SLURM

## Preliminary Results

| Accuracy | SHAP Stability |
|---|---|
| 0.795 | 0.876770 |
| 0.765 | 0.930213 |
| 0.775 | 0.901509 |

The experiment identified three nondominated configurations with different accuracy–stability trade-offs.

These are preliminary results and require further evaluation.

## Reproducing the Experiment

Install compatible versions of PyTorch, torchvision, datasets, SHAP, pymoo, NumPy, SciPy, pandas, and matplotlib.

Download the Hugging Face dataset `dragonintelligence/CIFAKE-image-dataset` and save it locally as `cifake_dataset`.

Run:

```bash
python run_nsga2_acenet.py --dataset-path cifake_dataset --pop-size 6 --generations 3 --output nsga2_results.csv
```

On a compatible SLURM cluster, submit `run_experiment.sh`.

## Future Work

- Evaluate more random seeds and larger validation samples
- Increase NSGA-II population size and generations
- Evaluate the selected models on untouched test data
- Compare results against baseline hyperparameter search methods
- Investigate alternative SHAP stability measurements
- Examine whether optimization should extend beyond hyperparameters

## Research Status

Initial proof-of-concept experiment completed successfully. Further validation and experimental design are in progress.
