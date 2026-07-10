import pandas as pd
from scipy.stats import pearsonr
import numpy as np
import torch
import matplotlib.pyplot as plt
from collectdata import obs_to_vector, collect_dataset
from blockPush import BlockPushEnv
from attribution import color_flip_test
from push_policy import train_bc 

# def rollout_policy(env, model, max_steps=15, fixed_distance=0.3, fixed_speed=0.5):
#     obs = env.reset()
#     for _ in range(max_steps):
#         x = obs_to_vector(obs)
#         with torch.no_grad():
#             angle = model(torch.tensor(x).unsqueeze(0)).item()
#         obs, reward, done, _ = env.step([angle, fixed_distance, fixed_speed])
#         if done:
#             break
#     # final distance to goal (lower = better)
#     block_pos = obs["block_pos"]
#     goal_pos = obs["goal_pos"]
#     final_dist = np.linalg.norm(np.array(block_pos[:2]) - np.array(goal_pos[:2]))
#     return final_dist


# def evaluate_policy(model, spurious_correlated, bias_strength, n_episodes=100):
#     env = BlockPushEnv(gui=False, randomize=True,
#                         spurious_correlated=spurious_correlated,
#                         bias_strength=bias_strength)
#     dists = [rollout_policy(env, model) for _ in range(n_episodes)]
#     env.close()
#     return np.mean(dists)

def rollout_policy_singlestep(env, model, fixed_distance=0.3, fixed_speed=0.5):
    obs = env.reset()
    x = obs_to_vector(obs)
    with torch.no_grad():
        angle = model(torch.tensor(x).unsqueeze(0)).item()
    obs, reward, done, _ = env.step([angle, fixed_distance, fixed_speed])
    block_pos = obs["block_pos"]
    goal_pos = obs["goal_pos"]
    return np.linalg.norm(np.array(block_pos[:2]) - np.array(goal_pos[:2]))

def evaluate_policy_v2(model, spurious_correlated, bias_strength, n_episodes=100, base_seed=1000):
    env = BlockPushEnv(gui=False, randomize=True,
                        spurious_correlated=spurious_correlated,
                        bias_strength=bias_strength)
    single_dists, multi_dists = [], []
    for i in range(n_episodes):
        np.random.seed(base_seed + i)  # decouple eval scenarios from training RNG state
        obs = env.reset()
        # single-step
        x = obs_to_vector(obs)
        with torch.no_grad():
            angle = model(torch.tensor(x).unsqueeze(0)).item()
        obs_s, _, _, _ = env.step([angle, 0.3, 0.5])
        single_dists.append(np.linalg.norm(np.array(obs_s["block_pos"][:2]) - np.array(obs_s["goal_pos"][:2])))
        # multi-step (reuse same starting scenario would require reset logic; simplest is separate rollout)
        multi_dists.append(rollout_policy_singlestep(env, model))
    env.close()
    return np.mean(single_dists), np.mean(multi_dists)  # return both single-step and multi-step distances

def compute_reliance(model, X):
    delta = color_flip_test(model, X)
    return delta.mean()

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    # Bias levels are split upfront. The calibration set is used to build the
    # reliance→drop regressor; the holdout set is used to test genuine prediction:
    # OOD drop is forecast from reliance alone and only revealed afterward.
    CALIBRATION_BIASES = [0.6, 0.7, 0.8, 0.9]
    HOLDOUT_BIASES     = [0.95, 0.99]
    N_SEEDS            = 3

    # ── Shared helper ────────────────────────────────────────────────────────
    def run_training_and_reliance(bias, seed):
        """Train a BC model and return (model, X, reliance). No OOD rollouts."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = BlockPushEnv(gui=False, randomize=True,
                           spurious_correlated=True, bias_strength=bias)
        X, y = collect_dataset(env, n_samples=2000)
        model, _ = train_bc(X, y, epochs=200)
        env.close()
        return model, X, compute_reliance(model, X)

    # ── Phase 1: Calibration ─────────────────────────────────────────────────
    # Train models, measure reliance AND pay the expensive OOD evaluation cost.
    # These paired observations are used to fit the reliance → drop regressor.
    print("=== Phase 1: Calibration (reliance + OOD evaluation) ===")
    cal_rows = []
    for bias in CALIBRATION_BIASES:
        for seed in range(N_SEEDS):
            model, X, reliance = run_training_and_reliance(bias, seed)
            in_dist, _ = evaluate_policy_v2(model, spurious_correlated=True,  bias_strength=bias)
            ood,     _ = evaluate_policy_v2(model, spurious_correlated=False, bias_strength=bias)
            drop = ood - in_dist
            cal_rows.append(dict(bias=bias, seed=seed, reliance=reliance,
                                 in_dist=in_dist, ood=ood, drop=drop))
            print(f"  bias={bias:.2f} seed={seed}  reliance={reliance:.4f}  "
                  f"in_dist={in_dist:.4f}  ood={ood:.4f}  drop={drop:+.4f}", flush=True)

    df_cal = pd.DataFrame(cal_rows)

    # Fit linear regressor on calibration data: reliance → OOD drop
    reg = LinearRegression()
    reg.fit(df_cal[["reliance"]], df_cal["drop"])
    cal_r, cal_p = pearsonr(df_cal["reliance"], df_cal["drop"])
    print(f"\nCalibration  Pearson r={cal_r:.3f}  p={cal_p:.4f}")
    print(f"Regressor:  drop ≈ {reg.coef_[0]:.4f} × reliance + {reg.intercept_:.4f}")

    # ── Phase 2: Holdout prediction (offline only — no OOD rollouts yet) ─────
    # For each holdout model, measure reliance then lock in a prediction.
    # The actual OOD cost is NOT paid at this stage.
    print("\n=== Phase 2: Holdout — predictions committed before OOD evaluation ===")
    holdout_rows = []
    holdout_models = []   # kept so Phase 3 can reuse them without retraining
    for bias in HOLDOUT_BIASES:
        for seed in range(N_SEEDS):
            model, X, reliance = run_training_and_reliance(bias, seed)
            predicted_drop = float(reg.predict([[reliance]])[0])
            print(f"  bias={bias:.2f} seed={seed}  reliance={reliance:.4f}  "
                  f"predicted_drop={predicted_drop:+.4f}  (OOD rollout not yet run)")
            holdout_rows.append(dict(bias=bias, seed=seed, reliance=reliance,
                                     predicted_drop=predicted_drop))
            holdout_models.append((bias, seed, model))

    # ── Phase 3: Reveal actual OOD drop on holdout set ───────────────────────
    # Now pay the expensive evaluation cost and compare against the predictions
    # that were already committed in Phase 2.
    print("\n=== Phase 3: Reveal — actual OOD evaluation on holdout set ===")
    for row, (bias, seed, model) in zip(holdout_rows, holdout_models):
        in_dist, _ = evaluate_policy_v2(model, spurious_correlated=True,  bias_strength=bias)
        ood,     _ = evaluate_policy_v2(model, spurious_correlated=False, bias_strength=bias)
        row["in_dist"]     = in_dist
        row["ood"]         = ood
        row["actual_drop"] = ood - in_dist
        row["error"]       = row["predicted_drop"] - row["actual_drop"]
        print(f"  bias={bias:.2f} seed={seed}  predicted={row['predicted_drop']:+.4f}  "
              f"actual={row['actual_drop']:+.4f}  error={row['error']:+.4f}")

    df_holdout = pd.DataFrame(holdout_rows)
    mae = mean_absolute_error(df_holdout["actual_drop"], df_holdout["predicted_drop"])
    ho_r, ho_p = pearsonr(df_holdout["reliance"], df_holdout["actual_drop"])
    print(f"\nHoldout  MAE={mae:.4f}  Pearson r (reliance vs actual_drop)={ho_r:.3f}  p={ho_p:.4f}")

    # ── Save results ─────────────────────────────────────────────────────────
    df_cal.to_csv("ood_calibration.csv", index=False)
    df_holdout.to_csv("ood_holdout_predictions.csv", index=False)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: calibration scatter + fitted line
    ax = axes[0]
    sc = ax.scatter(df_cal["reliance"], df_cal["drop"],
                    c=df_cal["bias"], cmap="viridis", s=60, zorder=3)
    plt.colorbar(sc, ax=ax, label="bias_strength")
    x_range = np.linspace(df_cal["reliance"].min(), df_cal["reliance"].max(), 100)
    ax.plot(x_range, reg.predict(x_range.reshape(-1, 1)),
            color="black", lw=1.5, ls="--", label=f"fit  r={cal_r:.2f}")
    ax.set_xlabel("reliance (mean |Δangle| from color flip)")
    ax.set_ylabel("OOD drop (m)")
    ax.set_title("Phase 1: Calibration — fitting reliance → drop")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: holdout predicted vs actual
    ax = axes[1]
    ax.scatter(df_holdout["actual_drop"], df_holdout["predicted_drop"],
               c=df_holdout["bias"], cmap="plasma", s=80, zorder=3)
    lims = [min(df_holdout[["actual_drop", "predicted_drop"]].min()),
            max(df_holdout[["actual_drop", "predicted_drop"]].max())]
    ax.plot(lims, lims, "k--", lw=1, label="perfect prediction")
    ax.set_xlabel("actual OOD drop (m)  [revealed in Phase 3]")
    ax.set_ylabel("predicted OOD drop (m)  [committed in Phase 2]")
    ax.set_title(f"Phase 2→3: Early-warning prediction\nMAE={mae:.4f}  r={ho_r:.2f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Interpretability signal as early warning: reliance predicts OOD failure\n"
        "without running OOD rollouts on the held-out bias levels",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig("early_warning_prediction.png", dpi=120)
    plt.show()
    print("Saved early_warning_prediction.png")