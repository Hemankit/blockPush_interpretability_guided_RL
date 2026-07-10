import torch
import numpy as np
import pandas as pd
from collectdata import obs_to_vector
from blockPush import BlockPushEnv
from push_policy import PushPolicy
def paired_color_rollout(env, model, n_pairs=100, max_steps=15,
                          fixed_distance=0.3, fixed_speed=0.5, base_seed=2000):
    """
    For each of n_pairs physical scenarios, run TWO full rollouts from
    identical starting conditions -- one with color_marker=0, one with
    color_marker=1. Everything else (block/goal position, mass, friction,
    size) is held fixed across the pair. Any difference in outcome is
    attributable ONLY to color_marker.
    """
    results = []

    for i in range(n_pairs):
        rng = np.random.default_rng(base_seed + i)
        initial_position = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.5]
        goal_pos = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.5]

        base_state = {
            "block_mass": 1.0,
            "block_friction": 0.5,
            "block_size": 1.0,
            "initial_position": initial_position,
            "goal_pos": goal_pos,
        }

        final_positions = {}
        for color in (0.0, 1.0):
            state = dict(base_state, color_marker=color)
            obs = env.reset(force_state=state)

            for _ in range(max_steps):
                x = obs_to_vector(obs)
                with torch.no_grad():
                    angle = model(torch.tensor(x).unsqueeze(0)).item()
                obs, reward, done, _ = env.step([angle, fixed_distance, fixed_speed])
                if done:
                    break

            final_positions[color] = np.array(obs["block_pos"][:2])

        divergence = np.linalg.norm(final_positions[0.0] - final_positions[1.0])
        dist_to_goal_0 = np.linalg.norm(final_positions[0.0] - np.array(goal_pos[:2]))
        dist_to_goal_1 = np.linalg.norm(final_positions[1.0] - np.array(goal_pos[:2]))

        results.append({
            "pair_idx": i,
            "divergence": divergence,           # how far apart the two final positions are
            "dist_goal_color0": dist_to_goal_0,
            "dist_goal_color1": dist_to_goal_1,
            "goal_dist_diff": abs(dist_to_goal_0 - dist_to_goal_1),
        })

    return pd.DataFrame(results)