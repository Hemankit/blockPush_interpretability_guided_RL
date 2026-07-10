"""Set up for a block pushing task"""
import pybullet as p
import pybullet_data
import math
import numpy as np
import os

class BlockPushEnv:
    def __init__(self, gui=False, randomize=False, spurious_correlated=False, bias_strength=0.9):
        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)

        self.randomize = randomize
        self.goal_pos = None
        self.block_id = None
        self.pusher_id = None
        self.color_marker = None
        self.spurious_correlated = spurious_correlated
        self.bias_strength = bias_strength

    def reset(self, force_state=None):
        if force_state is not None:
            block_mass = force_state.get("block_mass", 1.0)
            block_friction = force_state.get("block_friction", 0.5)
            block_size = force_state.get("block_size", 1.0)
            initial_position = force_state["initial_position"]
            goal_pos = force_state["goal_pos"]
            color_marker = force_state["color_marker"]
        else:
            block_mass, block_friction, block_size, initial_position, goal_pos, color_marker = self.randomize_variables()

        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)

        self.goal_pos = goal_pos
        self.color_marker = color_marker
        self.pusher_id = None

        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.table = p.loadURDF(
        "table/table.urdf",
        basePosition=[0, 0, 0],
        physicsClientId=self.client
    )

        self.block_id = p.loadURDF(
        "block.urdf",
        basePosition=initial_position,
        globalScaling=block_size,
        physicsClientId=self.client
    )

        p.changeDynamics(
        self.block_id,
        -1,
        mass=block_mass,
        lateralFriction=block_friction,
        physicsClientId=self.client
    )

        return self.get_observation()

    def step(self, action):
        push_angle, push_distance, push_speed = action
        self.apply_push(push_angle, push_distance, push_speed)

    # enough substeps to close the gap, plus buffer for post-contact push
        timestep = 1.0 / 240.0
        travel_steps = int(math.ceil((push_distance / push_speed) / timestep))
        buffer_steps = 60   # extra time to actually shove the block after contact
        total_substeps = travel_steps + buffer_steps

        for _ in range(total_substeps):
            p.stepSimulation(physicsClientId=self.client)

        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.is_done()
        return obs, reward, done, {}

    def apply_push(self, push_angle, push_distance, push_speed=0.5):
      block_pos, _ = p.getBasePositionAndOrientation(
      self.block_id,
        physicsClientId=self.client
    )
      start_x = block_pos[0] - push_distance * math.cos(push_angle)
      start_y = block_pos[1] - push_distance * math.sin(push_angle)
      if self.pusher_id is None:
          _pusher_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pusher", "pusher.urdf")
          self.pusher_id = p.loadURDF(
              _pusher_urdf,
              basePosition=[start_x, start_y, 0.5],
              physicsClientId=self.client
          )
      else:
          p.resetBasePositionAndOrientation(
              self.pusher_id,
              [start_x, start_y, 0.5],
              [0, 0, 0, 1],
              physicsClientId=self.client
          )
      vx = push_speed * math.cos(push_angle)
      vy = push_speed * math.sin(push_angle)
      p.resetBaseVelocity(
            self.pusher_id,
            linearVelocity=[vx, vy, 0],
            physicsClientId=self.client
        )
      
    def get_observation(self):
        block_pos, _ = p.getBasePositionAndOrientation(
            self.block_id,
            physicsClientId=self.client
        )

        if self.pusher_id is not None:
            pusher_pos, _ = p.getBasePositionAndOrientation(
                self.pusher_id,
                physicsClientId=self.client
            )
        else:
            pusher_pos = [0, 0, 0.5]

        return {
            "block_pos": block_pos,
            "pusher_pos": pusher_pos,
            "goal_pos": self.goal_pos,
            "color_marker": self.color_marker,
        }

    def compute_reward(self):
        block_pos, _ = p.getBasePositionAndOrientation(
            self.block_id,
            physicsClientId=self.client
        )

        distance_to_goal = np.linalg.norm(
            np.array(block_pos[:2]) - np.array(self.goal_pos[:2])
        )

        return -distance_to_goal

    def is_done(self):
        block_pos, _ = p.getBasePositionAndOrientation(
            self.block_id,
            physicsClientId=self.client
        )

        distance_to_goal = np.linalg.norm(
            np.array(block_pos[:2]) - np.array(self.goal_pos[:2])
        )

        return distance_to_goal < 0.05

    def randomize_variables(self):
        if self.randomize:
            block_mass = np.random.uniform(0.5, 2.0)
            block_friction = np.random.uniform(0.1, 1.0)
            block_size = np.random.uniform(0.5, 1.5)
            initial_position = [
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                0.5
            ]
            goal_pos = [
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-1.0, 1.0),
                0.5
            ]
        else:
            block_mass = 1.0
            block_friction = 0.5
            block_size = 1.0
            initial_position = [0.0, 0.0, 0.5]
            goal_pos = [1.0, 0.0, 0.5]

             # --- spurious feature injection ---
    # color_marker is a synthetic, non-physical binary flag correlated
    # with goal side during training. It has no causal relationship to
    # reward or dynamics — it's threaded through observation only.
        goal_is_right = goal_pos[0] >= 0
        if self.spurious_correlated:
            # matches goal side with probability self.bias_strength
            # mismatches otherwise
            if np.random.rand() < self.bias_strength:
                color_marker = 1.0 if goal_is_right else 0.0
            else:
                color_marker = 0.0 if goal_is_right else 1.0
        else:
            color_marker = float(np.random.randint(0, 2))


        return block_mass, block_friction, block_size, initial_position, goal_pos, color_marker

    def close(self):
        p.disconnect(physicsClientId=self.client)
