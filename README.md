# BlockPush Interpretability-Guided RL

This repository studies shortcut learning in a simple PyBullet block-pushing task. A behavioral cloning policy is trained on demonstrations that contain a spurious binary feature, `color_marker`, which is correlated with the correct push direction during training but has no causal effect on the physics. The project asks two practical questions:

1. Can simple interpretability checks reveal that the policy is using the shortcut?
2. Can counterfactual data augmentation reduce that reliance and improve out-of-distribution performance?

## Why this repo exists

The environment is intentionally minimal so the shortcut is easy to control and measure.

- The expert policy uses only geometry: it pushes from the block toward the goal with `atan2(goal_y - block_y, goal_x - block_x)`.
- The learned policy sees an extra `color_marker` input that may be highly predictive during training.
- At test time, the spurious correlation can be removed to measure how badly the policy fails when the shortcut stops working.

That makes the repository a compact test bed for interpretability-guided robustness experiments.

## Repository overview

### Core environment and training

- `blockPush.py` defines the PyBullet environment and the spurious feature injection.
- `expert_script.py` contains the expert policy used to label demonstrations.
- `collectdata.py` converts observations into 7D feature vectors and collects training data.
- `push_policy.py` defines a small MLP and the behavioral cloning training loop.

### Analysis and evaluation

- `attribution.py` measures shortcut reliance with gradient sensitivity and direct color-flip perturbations.
- `pre_OOD_evaluation.py` evaluates in-distribution and OOD rollout performance.
- `paired_rollout.py` compares paired trajectories that differ only in `color_marker`.
- `three_way_comparison.py` sweeps bias strength and compares offline reliance, closed-loop divergence, and OOD drop.
- `counterfactual_comparison.py` compares baseline training against randomization and counterfactual augmentation.

### Utility and debugging scripts

- `blockpush_run.py` trains a model and saves basic sanity-check figures.
- `abalation_run.py` runs attribution-style checks on a saved model.
- `diagnosis.py` is a lightweight debugging script for stepping through behavior.
- `concentration_validation_tests/sham_pertubation_plot.py` runs a sham perturbation control.

### Included artifacts

The repository already contains generated outputs such as:

- trained weights in `trained_model.pt`
- a saved dataset in `train_data.npz`
- sweep tables in `sweep_results.csv`, `counterfactual_comparison.csv`, `ood_calibration.csv`, and `ood_holdout_predictions.csv`
- figures such as `sweep_three_way.png`, `counterfactual_comparison.png`, `early_warning_prediction.png`, and `training_loss.png`

## Task setup

Each observation is represented as:

```text
[block_x, block_y, pusher_x, pusher_y, goal_x, goal_y, color_marker]
```

The policy predicts a single scalar action: the push angle.

Important details:

- `color_marker` is binary and non-causal.
- `bias_strength` controls how strongly `color_marker` aligns with the goal side during training.
- OOD evaluation disables the correlation so `color_marker` is no longer informative.

## Methods used in the repo

### 1. Offline shortcut reliance

`color_flip_test` in `attribution.py` measures how much the predicted angle changes when `color_marker` is flipped while everything else stays fixed.

Large changes imply that the network is relying on the shortcut.

### 2. Closed-loop paired divergence

`paired_rollout.py` evaluates whether changing only the spurious feature leads to different behavior in the simulator. This is stricter than an offline perturbation because it measures downstream behavioral divergence.

### 3. OOD performance drop

`pre_OOD_evaluation.py` compares performance when the training correlation is present versus when it is removed. This is the main robustness target.

### 4. Counterfactual augmentation

`counterfactual_comparison.py` tests whether pairing each training sample with a color-flipped twin and the same label reduces shortcut use more effectively than naive randomization.

## Installation

This project is plain Python. A virtual environment is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch numpy matplotlib pandas scipy pybullet
```

If you already have a Python environment configured, installing the packages above is enough.

## Quick start

### Train a baseline model and generate sanity checks

```bash
python blockpush_run.py
```

This produces figures such as `atan2_sanity.png` and `training_loss.png`.

### Run attribution checks on a saved model

```bash
python abalation_run.py
```

### Run the three-metric sweep

```bash
python three_way_comparison.py
```

This script sweeps several `bias_strength` values and records:

- offline shortcut reliance
- closed-loop paired divergence
- OOD performance drop

Results are saved to `sweep_results.csv` and plotted in `sweep_three_way.png`.

### Compare counterfactual augmentation against baselines

```bash
python counterfactual_comparison.py
```

This evaluates four data conditions:

- `baseline`
- `blind_rand`
- `double_rand`
- `counterfactual`

Results are saved to `counterfactual_comparison.csv` and `counterfactual_comparison.png`.

### Run the early-warning OOD analysis

```bash
python pre_OOD_evaluation.py
```

This script fits a reliance-to-OOD-drop relationship on calibration bias levels, then tests whether reliance can predict OOD failure on held-out bias levels.

## Reading the outputs

The main quantities to pay attention to are:

- `reliance`: average change in predicted angle after flipping `color_marker`
- `closed_loop_divergence`: behavioral difference between paired rollouts
- `drop`: OOD distance minus in-distribution distance

Lower is generally better for all three.

## Notes

- This repository contains both source code and generated experiment artifacts.
- File names reflect the current repo state, including `abalation_run.py` and `sham_pertubation_plot.py`.
- The project appears intended for local experimentation rather than packaged distribution.

## Suggested citation context

If you use this code in a report or project write-up, describe it as a toy PyBullet benchmark for studying shortcut reliance in imitation learning with counterfactual interventions.