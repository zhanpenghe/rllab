from os import path

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc.overrides import overrides

ACTION_LIMIT = 10.0


class SawyerEnv(MujocoEnv, Serializable):

    FILE = path.join('sawyer', 'sawyer.xml')

    def __init__(self, *args, **kwargs):
        super(SawyerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.joint_angles(),
            self.gripper_to_goal(),
        ])

    @overrides
    @property
    def action_space(self):
        shape = self.model.actuator_ctrlrange[:, 0].shape
        lb = np.full(shape, -ACTION_LIMIT)
        ub = np.full(shape, ACTION_LIMIT)
        return spaces.Box(lb, ub)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        distance_to_go = self.gripper_to_goal_dist()
        vel_cost = 1e-2 * np.linalg.norm(self.joint_velocities())
        reward = -distance_to_go - vel_cost

        done = self.gripper_to_goal_dist() < self.goal_size()

        return Step(next_obs, reward, done)

    def joint_angles(self):
        return self.model.data.qpos.flat

    def joint_velocities(self):
        return self.model.data.qvel.flat

    def gripper_to_goal(self):
        return self._get_geom_pos('rightclaw_it') - self._get_geom_pos(
            'goalbox')

    def gripper_to_goal_dist(self):
        return np.linalg.norm(self.gripper_to_goal())

    def goal_size(self):
        return self._get_geom_size('goalbox')

    def _get_geom_pos(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.data.geom_xpos[idx]

    def _get_geom_size(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.geom_size[idx][0]
