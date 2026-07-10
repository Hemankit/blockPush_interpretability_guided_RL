
from blockPush import BlockPushEnv


env = BlockPushEnv(gui=False, randomize=True)
obs = env.reset(force_state={
    "initial_position": [0.0, 0.0, 0.5],
    "goal_pos": [1.0, 0.0, 0.5],
    "color_marker": 0.0,
})
print("start block_pos:", obs["block_pos"])

for i in range(15):
    obs, r, done, _ = env.step([0.0, 0.3, 0.5])
    print(f"step {i}: block_pos = {obs['block_pos']}, reward = {r:.4f}")

env.close()