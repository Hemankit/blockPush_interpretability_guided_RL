# BlockPush Interpretability-Guided RL

A research project studying **shortcut learning** in imitation learning policies, using a PyBullet block-pushing environment. The core question: when a neural network policy is trained on data with spurious correlations, can interpretability tools detect the shortcut — and can counterfactual data augmentation fix it?

---

## Overview

A pusher robot must push a block to a goal position. The training data includes a `color_marker` feature that is **spuriously correlated** with the push direction (controlled by `bias_strength`). The ground-truth expert policy deliberately ignores this feature, but a naively trained behavioral cloning policy may learn to rely on it as a shortcut.

This repo investigates:
1. **Detection** – Using gradient sensitivity and counterfactual flip tests to measure how much a policy relies on the spurious `color_marker`.
2. **Mitigation** – Comparing counterfactual data augmentation against baseline and randomization strategies.
3. **Evaluation** – Measuring OOD performance drop when the spurious correlation is absent at test time.

---

## Environment

The environment (`blockPush.py`) is a PyBullet physics simulation:

- **State vector** (7 features): `[block_x, block_y, pusher_x, pusher_y, goal_x, goal_y, color_marker]`
- **Action**: push angle (radians), computed as `atan2(goal_y - block_y, goal_x - block_x)`
- **Spurious feature**: `color_marker` ∈ {0, 1} — binary, correlated with push direction during training at rate `bias_strength`, but causally irrelevant
- **OOD test**: evaluate with `spurious_correlated=False` so `color_marker` is no longer predictive

---

## Project Structure

```
blockPush_interpretability_guided_RL/
├── blockPush.py                  # PyBullet block-push environment
├── expert_script.py              # Ground-truth expert policy (ignores color_marker)
├── collectdata.py                # Dataset collection: obs → feature vector
├── push_policy.py                # MLP policy + behavioral cloning (BC) training
├── attribution.py                # Interpretability: gradient sensitivity & color-flip ablation
│
├── blockpush_run.py              # Full training run with atan2 sanity check & loss plots
├── abalation_run.py              # Load saved model; run both attribution checks
├── diagnosis.py                  # Environment step-through for debugging
│
├── pre_OOD_evaluation.py         # In-distribution vs OOD policy evaluation
├── paired_rollout.py             # Closed-loop paired rollouts (color 0 vs 1, same physics)
├── three_way_comparison.py       # Bias-strength sweep: reliance, divergence, OOD drop
├── counterfactual_comparison.py  # Four training conditions: baseline / blind_rand / double_rand / counterfactual
│
├── trained_model.pt              # Saved model weights
├── train_data.npz                # Saved training dataset
├── sweep_results.csv             # Results from three_way_comparison sweep
├── counterfactual_comparison.csv # Results from counterfactual experiment
├── ood_calibration.csv           # OOD calibration data
│
└── concentration_validation_tests/
    └── sham_pertubation_plot.py  # Sham perturbation control (pusher_pos noise vs color-flip)
```

---

## Key Components

### Expert Policy
The expert (`expert_script.py`) computes the optimal push angle using only block and goal positions:
```python
push_angle = atan2(goal_y - block_y, goal_x - block_x)
```
It **never** uses `color_marker`. Any policy that learns to use it has learned a shortcut.

### Policy Network
A 2-layer MLP (`push_policy.py`) trained via MSE behavioral cloning:
- Input: 7-feature state vector
- Output: push angle (scalar)
- Hidden size: 32, activation: ReLU

### Interpretability Tools (`attribution.py`)

| Tool | Method |
|------|--------|
| `color_marker_sensitivity` | Gradient `∂angle/∂color_marker` via backprop |
| `color_flip_test` | `|f(x) - f(x with color flipped)|` — direct causal perturbation |

### Counterfactual Augmentation (`counterfactual_comparison.py`)

Four training conditions compared at each bias level:

| Condition | Description | Dataset size |
|-----------|-------------|--------------|
| `baseline` | Raw correlated data | N |
| `blind_rand` | `color_marker` replaced with Bernoulli(0.5) | N |
| `double_rand` | Original + randomized copy, same labels | 2N |
| `counterfactual` | Each sample paired with its color-flipped twin | 2N |

`counterfactual` vs `double_rand` isolates the benefit of **causal pairing** beyond dataset size.

---

## Running the Code

### 1. Install dependencies
```bash
pip install torch numpy matplotlib pandas scipy pybullet
```

### 2. Train a model and run sanity checks
```bash
python blockpush_run.py
```
Generates `atan2_sanity.png` and training loss plots.

### 3. Run attribution ablation on saved model
```bash
python abalation_run.py
```
Prints gradient magnitudes and color-flip statistics; saves scatter plots.

### 4. Bias-strength sweep (three-way metrics)
```bash
python three_way_comparison.py
```
Sweeps `bias_strength` ∈ {0.6, 0.7, 0.8, 0.9, 0.95, 0.99} across 3 seeds.
Measures offline reliance, closed-loop behavioral divergence, and OOD performance drop.
Saves results to `sweep_results.csv`.

### 5. Counterfactual augmentation comparison
```bash
python counterfactual_comparison.py
```
Trains four models per bias level and saves results to `counterfactual_comparison.csv`.

### 6. Sham perturbation validation
```bash
python concentration_validation_tests/sham_pertubation_plot.py
```
Confirms that `color_flip_test` signal is not an artifact — sham noise on `pusher_pos` produces near-zero response.

---

## Key Metrics

- **Offline reliance**: Mean `|Δangle|` from flipping `color_marker` on the training set
- **Closed-loop divergence**: Mean final-position distance between paired rollouts (color 0 vs 1, same physics)
- **OOD drop**: `mean_dist_OOD − mean_dist_in_dist` — positive = worse OOD performance, indicating shortcut use

---

## Results Summary

Higher `bias_strength` → stronger spurious correlation → higher shortcut reliance → larger OOD performance drop. Counterfactual augmentation reduces reliance by forcing the model to produce the same output regardless of `color_marker`, directly breaking the shortcut during training.

---

## Dependencies

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [PyBullet](https://pybullet.org/)
- NumPy, Matplotlib, Pandas, SciPy
