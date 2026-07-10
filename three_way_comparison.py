import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from blockPush import BlockPushEnv
from push_policy import train_bc
from collectdata import collect_dataset
from paired_rollout import paired_color_rollout
from attribution import color_flip_test
from OOD_evaluation import evaluate_policy_v2

bias_strengths = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
n_seeds = 3
n_pairs_per_model = 100
n_ood_episodes = 100

sweep_results = []

for bias in bias_strengths:
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- train ---
        train_env = BlockPushEnv(gui=False, randomize=True,
                                  spurious_correlated=True, bias_strength=bias)
        X, y = collect_dataset(train_env, n_samples=2000)
        model, _ = train_bc(X, y, epochs=200)
        train_env.close()

        # --- metric 1: offline reliance (static, already validated) ---
        offline_reliance = color_flip_test(model, X).mean()

        # --- metric 2: closed-loop paired divergence (now physics-corrected) ---
        pair_env = BlockPushEnv(gui=False, randomize=True)
        paired_df = paired_color_rollout(pair_env, model,
                                          n_pairs=n_pairs_per_model,
                                          base_seed=3000 + seed * 1000)
        pair_env.close()
        closed_loop_divergence = paired_df["divergence"].mean()

        # --- metric 3: corrected in-dist vs OOD task performance ---
        in_dist_perf, _ = evaluate_policy_v2(model, spurious_correlated=True,
                                             bias_strength=bias, n_episodes=n_ood_episodes)
        ood_perf, _ = evaluate_policy_v2(model, spurious_correlated=False,
                                         bias_strength=bias, n_episodes=n_ood_episodes)
        drop = ood_perf - in_dist_perf

        sweep_results.append({
            "bias_strength": bias,
            "seed": seed,
            "offline_reliance": offline_reliance,
            "closed_loop_divergence": closed_loop_divergence,
            "in_dist_perf": in_dist_perf,
            "ood_perf": ood_perf,
            "drop": drop,
        })

        print(f"bias={bias:.2f} seed={seed}  "
              f"offline_reliance={offline_reliance:.4f}  "
              f"closed_loop_div={closed_loop_divergence:.4f}  "
              f"in_dist={in_dist_perf:.4f}  ood={ood_perf:.4f}  drop={drop:.4f}")

df = pd.DataFrame(sweep_results)
df.to_csv("sweep_results.csv", index=False)

print("\n=== Correlations across the sweep ===")
r1, p1 = pearsonr(df["offline_reliance"], df["closed_loop_divergence"])
print(f"offline_reliance vs closed_loop_divergence : r={r1:.3f}, p={p1:.4f}")

r2, p2 = pearsonr(df["offline_reliance"], df["drop"])
print(f"offline_reliance vs OOD drop                : r={r2:.3f}, p={p2:.4f}")

r3, p3 = pearsonr(df["closed_loop_divergence"], df["drop"])
print(f"closed_loop_divergence vs OOD drop           : r={r3:.3f}, p={p3:.4f}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

pairs = [
    ("offline_reliance", "closed_loop_divergence", axes[0]),
    ("offline_reliance", "drop", axes[1]),
    ("closed_loop_divergence", "drop", axes[2]),
]

for xcol, ycol, ax in pairs:
    sc = ax.scatter(df[xcol], df[ycol], c=df["bias_strength"], cmap="viridis", s=60)
    r, p = pearsonr(df[xcol], df[ycol])
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"r={r:.2f}, p={p:.3f}")

fig.colorbar(sc, ax=axes, label="bias_strength", shrink=0.8)
plt.savefig("sweep_three_way.png", dpi=120, bbox_inches="tight")
plt.show()