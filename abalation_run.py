import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from attribution import color_flip_test, color_marker_sensitivity
from blockPush import BlockPushEnv
from collectdata import collect_dataset
from push_policy import PushPolicy, train_bc

MODEL_PATH = "trained_model.pt"
DATA_PATH  = "train_data.npz"

# --- 1. Train once, then reuse the saved model ---
if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
    print("Loading saved model and data...")
    data  = np.load(DATA_PATH)
    X, y  = data["X"], data["y"]
    model = PushPolicy(input_dim=X.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
else:
    env = BlockPushEnv(gui=False, randomize=True,
                       spurious_correlated=True, bias_strength=0.9)
    X, y = collect_dataset(env, n_samples=2000)
    env.close()

    model, _ = train_bc(X, y)

    torch.save(model.state_dict(), MODEL_PATH)
    np.savez(DATA_PATH, X=X, y=y)
    print(f"Model saved to {MODEL_PATH}, data saved to {DATA_PATH}")

# --- 2. Run both attribution checks on the SAME training-distribution data ---
grads = color_marker_sensitivity(model, X)
delta = color_flip_test(model, X)

# --- 3. Compare color_marker against goal_pos as a sanity baseline ---
X_t = torch.tensor(X, requires_grad=True)
pred = model(X_t)
pred.sum().backward()
all_grads = X_t.grad.numpy()

print("=== Gradient magnitudes (mean |∂angle/∂feature|) ===")
print(f"goal_x       : {np.abs(all_grads[:,4]).mean():.5f}")
print(f"goal_y       : {np.abs(all_grads[:,5]).mean():.5f}")
print(f"color_marker : {np.abs(all_grads[:,6]).mean():.5f}")

print("\n=== Color-flip ablation ===")
print(f"mean |Δangle| from flipping color_marker : {delta.mean():.5f} rad "
      f"({np.degrees(delta.mean()):.2f}°)")
print(f"max  |Δangle|                              : {delta.max():.5f} rad "
      f"({np.degrees(delta.max()):.2f}°)")
print(f"fraction of samples with |Δangle| > 1°     : {(np.degrees(delta) > 1).mean():.2%}")

# --- 4. Scatter: goal position coloured by color-flip sensitivity ---
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(X[:, 4], X[:, 5], c=np.degrees(delta),
                cmap="hot", s=8, alpha=0.5, vmin=0)
plt.colorbar(sc, ax=ax, label="|Δangle| from color flip (°)")
ax.set_xlabel("goal_x")
ax.set_ylabel("goal_y")
ax.set_title("Color-flip sensitivity by goal position")
plt.tight_layout()
plt.savefig("color_flip_sensitivity.png", dpi=150)
plt.show()
print("Plot saved to color_flip_sensitivity.png")