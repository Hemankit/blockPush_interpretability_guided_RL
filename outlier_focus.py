import torch
import numpy as np
from blockPush import BlockPushEnv
from collectdata import obs_to_vector
from push_policy import PushPolicy

MODEL_PATH = "trained_model.pt"
DATA_PATH  = "train_data.npz"

data  = np.load(DATA_PATH)
X     = data["X"]
model = PushPolicy(input_dim=X.shape[1])
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
def trace_pair(env, model, initial_position, goal_pos, max_steps=15,
                fixed_distance=0.3, fixed_speed=0.5):
    traces = {}
    for color in (0.0, 1.0):
        state = {
            "block_mass": 1.0, "block_friction": 0.5, "block_size": 1.0,
            "initial_position": initial_position, "goal_pos": goal_pos,
            "color_marker": color
        }
        obs = env.reset(force_state=state)
        path = [obs["block_pos"][:2]]
        for _ in range(max_steps):
            x = obs_to_vector(obs)
            with torch.no_grad():
                angle = model(torch.tensor(x).unsqueeze(0)).item()
            obs, r, done, _ = env.step([angle, fixed_distance, fixed_speed])
            path.append(obs["block_pos"][:2])
            if done:
                break
        traces[color] = path
    return traces

# reconstruct scenario 94 using the same base_seed logic as paired_color_rollout
rng = np.random.default_rng(2000 + 94)
initial_position = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.5]
goal_pos = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.5]

env = BlockPushEnv(gui=False, randomize=True, spurious_correlated=True, bias_strength=0.9)
traces = trace_pair(env, model, initial_position, goal_pos)
env.close()

for color, path in traces.items():
    print(f"color={color}: {[tuple(round(c,3) for c in p) for p in path]}")