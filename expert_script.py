import math
import numpy as np

def expert_policy(obs):
    """
    Ground-truth expert: pushes the block directly toward goal_pos.
    Deliberately ignores color_marker -- the only input that matters
    is the vector from block to goal.
    """
    block_pos = obs["block_pos"]
    goal_pos = obs["goal_pos"]

    dx = goal_pos[0] - block_pos[0]
    dy = goal_pos[1] - block_pos[1]

    push_angle = math.atan2(dy, dx)
    return push_angle