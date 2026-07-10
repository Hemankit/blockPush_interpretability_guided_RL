
import numpy as np
import matplotlib.pyplot as plt
from blockPush import BlockPushEnv
from collectdata import collect_dataset
from push_policy import train_bc

env = BlockPushEnv(gui=False, randomize=True,
                    spurious_correlated=True, bias_strength=0.9)

X, y = collect_dataset(env, n_samples=2000)
model, loss_history = train_bc(X, y)

env.close()

# ── 1. atan2 sanity-check ────────────────────────────────────────────────────
# Convention: x right, y up (standard 2D Cartesian).
# Expert angle = atan2(goal_y - block_y, goal_x - block_x).
# Each arrow should point from the block directly toward the goal.
rng = np.random.default_rng(0)
idx = rng.choice(len(X), size=min(60, len(X)), replace=False)

bx, by = X[idx, 0], X[idx, 1]   # block x, y
gx, gy = X[idx, 4], X[idx, 5]   # goal  x, y
angles  = y[idx]                  # atan2(dy, dx) labels

# Arrow direction from atan2 label (unit length scaled for visibility)
scale = 0.15
ux = np.cos(angles) * scale
uy = np.sin(angles) * scale

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(bx, by, c="steelblue", s=30, zorder=3, label="block")
ax.scatter(gx, gy, c="tomato",    s=30, zorder=3, label="goal")
ax.quiver(bx, by, ux, uy,
          angles="xy", scale_units="xy", scale=1,
          color="steelblue", alpha=0.7, width=0.004,
          label="atan2 push dir")
# Thin line from block to goal to verify alignment
for i in range(len(idx)):
    ax.plot([bx[i], gx[i]], [by[i], gy[i]],
            color="gray", lw=0.4, alpha=0.4)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("atan2 sanity check\n(arrows should point block → goal)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("atan2_sanity.png", dpi=120)
plt.show()
print("Saved atan2_sanity.png")
true_angle = np.arctan2(gy - by, gx - bx)
angle_err = np.abs((angles - true_angle + np.pi) % (2*np.pi) - np.pi)  # wrap to [-pi,pi]
print(f"max angle error: {angle_err.max():.6f} rad, mean: {angle_err.mean():.6f} rad")

# ── 2. Training loss ─────────────────────────────────────────────────────────
# No label_noise on color_marker, so the expert is deterministic and MSE
# should converge close to zero if the network capacity is sufficient.
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(loss_history, color="steelblue", lw=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
ax.set_title("Behaviour-cloning training loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=120)
plt.show()
print("Saved training_loss.png")