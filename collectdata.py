import numpy as np
from expert_script import expert_policy
def obs_to_vector(obs):
    """
    Flatten obs dict into a fixed-order feature vector.
    Only x,y used for positions (z is constant table height / irrelevant).
    """
    bx, by, _ = obs["block_pos"]
    px, py, _ = obs["pusher_pos"]
    gx, gy, _ = obs["goal_pos"]
    c = obs["color_marker"]
    return np.array([bx, by, px, py, gx, gy, c], dtype=np.float32)


def collect_dataset(env, n_samples):
    X, y = [], []
    for _ in range(n_samples):
        obs = env.reset()
        action_angle = expert_policy(obs)
        X.append(obs_to_vector(obs))
        y.append(action_angle)
    return np.stack(X), np.array(y, dtype=np.float32)