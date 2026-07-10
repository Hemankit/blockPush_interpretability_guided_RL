"""
Counterfactual augmentation vs blind randomization
===================================================
Trains BC under four data conditions on the same correlated dataset and
compares shortcut reliance and OOD robustness:

  baseline       – raw correlated data, no intervention              (N samples)
  blind_rand     – color_marker replaced with uniform random bits    (N samples)
  double_rand    – original + a randomized copy, same labels         (2N samples)
  counterfactual – each sample paired with its color-flipped twin    (2N samples)

baseline vs blind_rand   → effect of removing the spurious correlation (N vs N)
double_rand vs blind_rand → effect of dataset size alone              (2N vs N)
counterfactual vs double_rand → effect of causal pairing structure    (2N vs 2N)

Reliance is measured via color_flip_test on the *original* correlated X so
the comparison is fair across conditions.
OOD drop = ood_dist - in_dist_dist  (positive = worse OOD, i.e. more shortcut use)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from blockPush import BlockPushEnv
from collectdata import collect_dataset
from push_policy import train_bc
from attribution import color_flip_test
from pre_OOD_evaluation import evaluate_policy_v2

# ── Augmentation helpers ─────────────────────────────────────────────────────

def counterfactual_augment(X, y):
    """Double the dataset: every sample paired with its color-flipped twin, same label."""
    X_flip = X.copy()
    X_flip[:, 6] = 1.0 - X_flip[:, 6]          # flip binary color_marker (index 6)
    return np.concatenate([X, X_flip]), np.concatenate([y, y])


def blind_randomize(X, y, rng):
    """Replace color_marker with i.i.d. Bernoulli(0.5), breaking the spurious correlation."""
    X_rand = X.copy()
    X_rand[:, 6] = rng.integers(0, 2, size=len(X)).astype(np.float32)
    return X_rand, y


def double_randomize(X, y, rng):
    """Double the dataset: original + a copy with color_marker randomized, same labels.

    Matches counterfactual's 2N size without the causal pairing structure, so
    counterfactual vs double_rand isolates pairing benefit from dataset-size benefit.
    """
    X_rand = X.copy()
    X_rand[:, 6] = rng.integers(0, 2, size=len(X)).astype(np.float32)
    return np.concatenate([X, X_rand]), np.concatenate([y, y])


# ── Experiment parameters ────────────────────────────────────────────────────

BIAS_STRENGTHS = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]  # match existing sweep scripts
N_SAMPLES      = 2000   # dataset size per seed
N_SEEDS        = 3      # independent repetitions per bias level
N_EPISODES     = 100    # episodes per OOD evaluation
EPOCHS         = 200    # BC training epochs

conditions = ["baseline", "blind_rand", "double_rand", "counterfactual"]
results    = []

# ── Main loop ────────────────────────────────────────────────────────────────

for bias in BIAS_STRENGTHS:
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        # Collect one correlated dataset; all three conditions share this base data
        env = BlockPushEnv(gui=False, randomize=True,
                           spurious_correlated=True, bias_strength=bias)
        X, y = collect_dataset(env, n_samples=N_SAMPLES)
        env.close()

        datasets = {
            "baseline":       (X,                                y),
            "blind_rand":     blind_randomize(X,       y, rng),
            "double_rand":    double_randomize(X,      y, rng),
            "counterfactual": counterfactual_augment(X, y),
        }

        for cond, (X_train, y_train) in datasets.items():
            print(f"\nbias={bias}  seed={seed}  condition={cond}  n_train={len(X_train)}")
            model, _ = train_bc(X_train, y_train, epochs=EPOCHS)

            # Reliance measured on original correlated X (fair comparison across conditions)
            reliance = color_flip_test(model, X).mean()

            in_dist, _ = evaluate_policy_v2(model, spurious_correlated=True,
                                            bias_strength=bias, n_episodes=N_EPISODES)
            ood, _     = evaluate_policy_v2(model, spurious_correlated=False,
                                            bias_strength=bias, n_episodes=N_EPISODES)
            drop = ood - in_dist   # positive → worse on OOD

            results.append(dict(bias=bias, seed=seed, condition=cond,
                                reliance=reliance, in_dist=in_dist, ood=ood, drop=drop))
            print(f"  reliance={reliance:.4f}  in_dist={in_dist:.4f}  "
                  f"ood={ood:.4f}  drop={drop:+.4f}")

# ── Summary table ────────────────────────────────────────────────────────────

df = pd.DataFrame(results)
df.to_csv("counterfactual_comparison.csv", index=False)

print("\n=== Summary (mean ± std across seeds, per bias level) ===")
summary = df.groupby(["bias", "condition"])[["reliance", "drop"]].agg(["mean", "std"])
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
print(summary.to_string(float_format="{:.4f}".format))

# ── Plot: reliance and OOD drop vs bias strength ─────────────────────────────

order  = ["baseline", "blind_rand", "double_rand", "counterfactual"]
colors = {"baseline": "steelblue", "blind_rand": "seagreen",
          "double_rand": "goldenrod", "counterfactual": "tomato"}
labels = {"baseline":       "Baseline (biased, N)",
          "blind_rand":     "Blind randomize (N)",
          "double_rand":    "Double randomize (2N)",
          "counterfactual": "Counterfactual augment (2N)"}

agg = df.groupby(["bias", "condition"])[["reliance", "drop"]].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, metric, ylabel, title in [
    (axes[0], "reliance", "mean |Δangle| from color flip (rad)",
     "Offline shortcut reliance vs bias strength\n(lower = less shortcut use)"),
    (axes[1], "drop",     "OOD dist − in-dist dist (m)",
     "OOD performance drop vs bias strength\n(lower = more robust)"),
]:
    for cond in order:
        sub = agg[agg.condition == cond].sort_values("bias")
        ax.plot(sub["bias"], sub[metric], marker="o", label=labels[cond],
                color=colors[cond], linewidth=2)
    ax.set_xlabel("Bias strength")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle(
    f"Counterfactual augmentation vs blind randomization across bias levels  "
    f"(n={N_SAMPLES}, seeds={N_SEEDS})\n"
    f"blind_rand vs double_rand isolates size effect; double_rand vs counterfactual isolates pairing",
    fontsize=10
)
plt.tight_layout()
plt.savefig("counterfactual_comparison.png", dpi=120)
plt.show()
print("Saved counterfactual_comparison.png")
