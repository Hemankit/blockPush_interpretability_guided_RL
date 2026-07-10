# Interpretability-Guided RL: Block-Push Shortcut Detection

This project investigates **spurious correlation / shortcut learning** in a behaviour-cloned
robot policy and demonstrates how interpretability tools can detect, quantify, and mitigate
the problem before deployment.

A simulated pusher robot must push a block to a goal position.  
During training a binary `color_marker` is artificially correlated with the correct push
direction, giving the policy an opportunity to use an irrelevant visual cue as a shortcut.
The pipeline then measures whether the policy exploited that shortcut, how badly it hurts
out-of-distribution (OOD) performance, and whether counterfactual data augmentation fixes it.

---

## Pipeline Overview

```
blockPush.py          ← 1. Environment
expert_script.py      ← 2. Expert oracle
collectdata.py        ← 3. Dataset collection
push_policy.py        ← 4. BC model definition
blockpush_run.py      ← 5. Train + sanity-check plots
abalation_run.py      ← 6. Reusable train / attribution runner
attribution.py        ← 7. Gradient & counterfactual attribution
diagnosis.py          ← 8. Environment smoke test
pre_OOD_evaluation.py ← 9. Calibrated OOD-drop prediction
counterfactual_comparison.py ← 10. Augmentation ablation
paired_rollout.py     ← 11. Closed-loop paired rollout
three_way_comparison.py      ← 12. Full metric sweep
concentration_validation_tests/sham_pertubation_plot.py ← 13. Sham control
```

---

## Step-by-Step Explanation

### Step 1 — `blockPush.py`: Physics Environment

Builds the PyBullet simulation of a table, a pushable block, and a goal marker.

**Key design choices:**
- `spurious_correlated=True` + `bias_strength` (0–1): when enabled, the environment
  biases the block's starting position so that `color_marker` is statistically predictive
  of the correct push direction at the chosen rate (e.g. 90 % of the time for `bias=0.9`).
- The physics parameters (mass, friction, block size) are randomised so the policy must
  generalise over them; `color_marker` is the *only* truly spurious feature.

**Why it matters:** The controlled spurious correlation lets us sweep over exactly *how
strong* the shortcut signal is and measure downstream harm with precision.

---

### Step 2 — `expert_script.py`: Ground-Truth Expert

Computes the optimal push angle as `atan2(goal_y − block_y, goal_x − block_x)`.

**Key design choice:** The expert *deliberately ignores* `color_marker`.  
Because `color_marker` is a pure spurious correlate in the training data, any reliance the
learned policy develops on it is a bug, not a feature.

**Why it matters:** This gives us a clean ground truth. We know the correct policy has
zero dependence on `color_marker`, so any non-zero sensitivity the trained model shows is
entirely a shortcut artefact.

---

### Step 3 — `collectdata.py`: Dataset Collection

Rolls out the expert policy across randomly initialised environments and stores
`(observation_vector, push_angle)` pairs.

The 7-dimensional feature vector is:

| Index | Feature      |
|-------|--------------|
| 0–1   | block_x, block_y |
| 2–3   | pusher_x, pusher_y |
| 4–5   | goal_x, goal_y |
| 6     | color_marker |

**Why it matters:** The feature vector deliberately includes `color_marker` alongside the
causally relevant position features. When `spurious_correlated=True`, `color_marker` has
predictive power at train time, creating the opportunity for shortcut learning.

---

### Step 4 — `push_policy.py`: Behaviour-Cloning Model

A small 2-layer ReLU MLP (`7 → 32 → 32 → 1`) trained with MSE loss to predict push angle
from the observation vector.

**Why it matters:** Behaviour cloning (BC) is a standard imitation-learning baseline and
is known to be susceptible to spurious correlations because it optimises raw prediction
accuracy on the training distribution without any causal reasoning. The architecture is
intentionally simple so that shortcut behaviour is easy to analyse.

---

### Step 5 — `blockpush_run.py`: Training + Sanity Checks

Orchestrates data collection and BC training, then produces two diagnostic plots:

1. **`atan2_sanity.png`** — Quiver plot confirming that the expert labels point from each
   block position toward the corresponding goal. Catches any label or coordinate-frame bugs
   before training.
2. **`training_loss.png`** — MSE training curve. Confirms the network converges, and that
   MSE is near zero (expected, because the expert is deterministic and `color_marker`
   carries no label noise).

**Why it matters:** Sanity checks at this stage prevent a "garbage in, garbage out"
failure. If the labels are wrong, all downstream attribution results will be meaningless.

---

### Step 6 — `abalation_run.py`: Persistent Train / Attribution Runner

Trains the model (or loads a saved `trained_model.pt` / `train_data.npz`) and runs
both attribution methods, printing gradient magnitudes for every feature and saving
`color_flip_sensitivity.png`.

**Why it matters:** Caching the trained model ensures that attribution experiments are
reproducible and that the ablation study always analyses the *same* model rather than
a freshly re-trained one with different random seed behaviour.

---

### Step 7 — `attribution.py`: Interpretability Methods

Two complementary attribution techniques:

| Method | What it measures |
|--------|-----------------|
| `color_marker_sensitivity` | Gradient `∂predicted_angle / ∂color_marker` — local, first-order sensitivity |
| `color_flip_test` | `|angle(color=0) − angle(color=1)|` — interventional, finite-difference effect |

**Why it matters:**  
- Gradient attribution is cheap and differentiable but can be misleading in saturated or
  noisy regions.  
- The color-flip test is a causal intervention: it directly asks "how much does the
  *output change* when I force the spurious feature to its opposite value?" This maps
  directly to real-world harm (a policy that changes its push direction based on an
  irrelevant colour will push the block to the wrong place whenever the colour changes).

---

### Step 8 — `diagnosis.py`: Environment Smoke Test

Resets the environment to a fixed state and steps through 15 physics steps, printing
block position and reward at each step.

**Why it matters:** A quick sanity check that the PyBullet simulation is wired up
correctly (collisions, reward function, URDF loading). Run this first if the environment
behaves unexpectedly.

---

### Step 9 — `pre_OOD_evaluation.py`: Calibrated OOD-Drop Prediction

Implements a two-phase procedure to predict OOD performance degradation from
interpretability scores *without* running a full OOD rollout for every model.

**Phase 1 — Calibration:** Train models at `bias ∈ {0.6, 0.7, 0.8, 0.9}`, measure both
`reliance` (from `color_flip_test`) and the actual OOD performance drop. Fit a
`LinearRegression` mapping reliance → OOD drop.

**Phase 2 — Holdout prediction:** Train models at unseen biases `{0.95, 0.99}`, measure
reliance only, and *forecast* the OOD drop without executing any OOD rollouts.

**Why it matters:** OOD evaluation requires running the policy in a new environment
distribution — expensive at scale. If interpretability scores reliably predict OOD drops,
they can serve as cheap *proxy safety metrics* during development, flagging high-risk
models before real-world deployment.

---

### Step 10 — `counterfactual_comparison.py`: Augmentation Ablation

Trains BC under four data conditions on the *same* base correlated dataset and compares
shortcut reliance and OOD robustness:

| Condition | Description | Size |
|-----------|-------------|------|
| `baseline` | Raw correlated data, no intervention | N |
| `blind_rand` | `color_marker` replaced with i.i.d. Bernoulli(0.5) | N |
| `double_rand` | Original + randomised copy, same labels | 2N |
| `counterfactual` | Every sample paired with its colour-flipped twin | 2N |

The comparisons are structured to isolate three effects:
- **baseline vs blind_rand**: effect of removing the spurious correlation
- **double_rand vs blind_rand**: effect of dataset size alone
- **counterfactual vs double_rand**: effect of *causal pairing structure* (same size)

**Why it matters:** Counterfactual augmentation is a principled fix rooted in causal
reasoning. By providing the model with matched pairs that differ *only* in `color_marker`
but share the same label, the gradient signal explicitly teaches the model that
`color_marker` carries no information. The ablation design rules out the simpler
explanation that "more data" is all that helps.

---

### Step 11 — `paired_rollout.py`: Closed-Loop Paired Divergence

For each of `n_pairs` scenarios, runs *two full rollouts* from identical starting
conditions — one with `color_marker=0`, one with `color_marker=1` — and records how far
apart the final block positions are (`divergence`).

**Why it matters:** Gradient and flip-test attributions are *static* (offline) — they
measure sensitivity on a fixed dataset without simulating the policy in a loop. The
paired divergence is a *dynamic* (online) metric: it measures causal impact through
actual closed-loop physics. A model with non-zero offline reliance but zero divergence
might be exploiting the shortcut in a way that cancels out in practice; divergence
confirms whether the shortcut has real physical consequences.

---

### Step 12 — `three_way_comparison.py`: Full Metric Sweep

Sweeps `bias_strength ∈ {0.6, 0.7, 0.8, 0.9, 0.95, 0.99}` with 3 random seeds each,
computing all three metrics per model:

1. **Offline reliance** (`color_flip_test` mean) — static attribution
2. **Closed-loop divergence** (`paired_color_rollout` mean) — dynamic physics test
3. **OOD drop** (`ood_perf − in_dist_perf`) — task performance degradation

Pearson correlations among the three metrics are printed and scatter-plotted to
`sweep_results.csv`.

**Why it matters:** This is the central validation of the project's thesis. If
interpretability-derived reliance scores truly predict deployment harm, they must
correlate with both the physics-based divergence and the task-level OOD drop. A high
correlation across the full sweep provides evidence that cheap offline attribution can
serve as a reliable safety signal.

---

### Step 13 — `concentration_validation_tests/sham_pertubation_plot.py`: Sham Control

Compares the color-flip effect against a *sham perturbation*: tiny Gaussian noise added
to `pusher_pos` (columns 2–3), a feature the expert also ignores.

Additionally compares the color-flip magnitude in two regions:
- **Low goal_x** (bottom 25th percentile) — where `color_marker` is most correlated with
  push direction in the biased training set
- **Rest of the space** — where the correlation is weaker

**Why it matters:**  
A good interpretability method must be *specific* — it should fire for the feature that
actually causes the behaviour, not for any random perturbation. By showing that:
1. The color-flip effect is many times larger than the sham-perturbation effect.
2. The effect is spatially concentrated in exactly the region where the spurious
   correlation was strongest during training.

...we validate that the attribution signal is meaningful, not an artefact of network
sensitivity to any change whatsoever.

---

## Running the Pipeline

```bash
# 1. Install dependencies
pip install pybullet torch numpy scipy matplotlib pandas scikit-learn

# 2. Smoke test the environment
python diagnosis.py

# 3. Train the base model and run sanity checks
python blockpush_run.py

# 4. Attribution ablation (loads saved model if present)
python abalation_run.py

# 5. Calibrated OOD-drop prediction
python pre_OOD_evaluation.py

# 6. Counterfactual augmentation comparison
python counterfactual_comparison.py

# 7. Full three-metric sweep
python three_way_comparison.py

# 8. Sham control / concentration validation
python concentration_validation_tests/sham_pertubation_plot.py
```

---

## Key Files

| File | Role |
|------|------|
| `blockPush.py` | PyBullet environment with controllable spurious correlation |
| `expert_script.py` | Ground-truth oracle (angle to goal, ignores color) |
| `collectdata.py` | Dataset collection + observation featurisation |
| `push_policy.py` | BC MLP definition and training loop |
| `blockpush_run.py` | End-to-end training + diagnostic plots |
| `abalation_run.py` | Cached train + attribution runner |
| `attribution.py` | Gradient sensitivity + color-flip attribution |
| `diagnosis.py` | Environment smoke test |
| `pre_OOD_evaluation.py` | Reliance → OOD drop calibration + holdout prediction |
| `counterfactual_comparison.py` | Augmentation strategy ablation |
| `paired_rollout.py` | Closed-loop paired divergence measurement |
| `three_way_comparison.py` | Full bias-level sweep across all three metrics |
| `concentration_validation_tests/sham_pertubation_plot.py` | Sham control + spatial concentration test |
| `trained_model.pt` | Saved BC model weights |
| `train_data.npz` | Saved training dataset |
| `sweep_results.csv` | Output of `three_way_comparison.py` |
| `ood_calibration.csv` | Calibration-phase results from `pre_OOD_evaluation.py` |
| `ood_holdout_predictions.csv` | Holdout-phase predictions from `pre_OOD_evaluation.py` |
| `counterfactual_comparison.csv` | Output of `counterfactual_comparison.py` |
