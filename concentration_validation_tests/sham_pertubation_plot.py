import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from push_policy import PushPolicy
from attribution import color_flip_test

# --- Load model and data ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "trained_model.pt")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "train_data.npz")

data = np.load(DATA_PATH)
X, y = data["X"], data["y"]

model = PushPolicy(input_dim=X.shape[1])
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

delta = color_flip_test(model, X)

# --- Sham perturbation: tiny noise on pusher_pos, which the expert also ignores ---
def sham_perturbation_test(model, X, noise_std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X_orig = X.copy()
    X_sham = X.copy()
    # pusher_pos is columns 2,3 -- perturb both slightly
    noise = rng.normal(0, noise_std, size=(X.shape[0], 2))
    X_sham[:, 2:4] += noise

    with torch.no_grad():
        pred_orig = model(torch.tensor(X_orig)).numpy()
        pred_sham = model(torch.tensor(X_sham)).numpy()

    return np.abs(pred_sham - pred_orig)

sham_delta = sham_perturbation_test(model, X)

# --- Split samples into "low goal_x" vs "rest" ---
goal_x = X[:, 4]
threshold = np.percentile(goal_x, 25)   # bottom quartile = "low goal_x"
low_mask = goal_x <= threshold
rest_mask = ~low_mask

print("=== Threshold ===")
print(f"low goal_x defined as goal_x <= {threshold:.3f}  (n={low_mask.sum()})")
print(f"rest: n={rest_mask.sum()}")

print("\n=== Color-flip delta (radians) ===")
print(f"low goal_x region : mean={delta[low_mask].mean():.5f}  median={np.median(delta[low_mask]):.5f}  max={delta[low_mask].max():.5f}")
print(f"rest of space      : mean={delta[rest_mask].mean():.5f}  median={np.median(delta[rest_mask]):.5f}  max={delta[rest_mask].max():.5f}")
print(f"ratio (low / rest) : {delta[low_mask].mean() / delta[rest_mask].mean():.2f}x")

print("\n=== Sham (pusher_pos noise) delta (radians) ===")
print(f"low goal_x region : mean={sham_delta[low_mask].mean():.5f}  median={np.median(sham_delta[low_mask]):.5f}  max={sham_delta[low_mask].max():.5f}")
print(f"rest of space      : mean={sham_delta[rest_mask].mean():.5f}  median={np.median(sham_delta[rest_mask]):.5f}  max={sham_delta[rest_mask].max():.5f}")
print(f"ratio (low / rest) : {sham_delta[low_mask].mean() / sham_delta[rest_mask].mean():.2f}x")

print("\n=== Direct comparison: color-flip vs sham, same region ===")
print(f"low goal_x  : color-flip mean={delta[low_mask].mean():.5f}  vs  sham mean={sham_delta[low_mask].mean():.5f}  (ratio {delta[low_mask].mean()/sham_delta[low_mask].mean():.1f}x)")
print(f"rest        : color-flip mean={delta[rest_mask].mean():.5f}  vs  sham mean={sham_delta[rest_mask].mean():.5f}  (ratio {delta[rest_mask].mean()/sham_delta[rest_mask].mean():.1f}x)")