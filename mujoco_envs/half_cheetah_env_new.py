from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import math
import os
import cv2
import gym
import dm_env
from gym import utils
from gym.spaces import Box
import numpy as np
from dm_env import StepType, specs
from typing import Any, NamedTuple
# from gym.envs.mujoco import mujoco_env
from mujoco_envs.mujoco_utils import MujocoTrait
# from gym.envs.mujoco import MuJocoPyEnv

"""
    ### Action Space
    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied between *links*.

    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
    | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
    | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
    | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
    | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
    | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |



    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
    | 3   | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angle (rad)              |
    | 4   | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
    | 5   | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
    | 6   | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angle (rad)              |
    | 7   | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
    | 8   | x-coordinate of the front tip        | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | y-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
    | 12  | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
    | 13  | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
    | 14  | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |

"""




def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    info: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        obs = self._env.reset()
        time_step=ExtendedTimeStep(observation=obs,
                                step_type=StepType.FIRST,
                                action=np.zeros(self._env._env.action_space.shape, dtype='float32'),
                                reward=0.0,
                                discount=1.0,
                                info={})
        return self._augment_time_step(time_step)

    def step(self, action, render=False):
        ob, reward, done, info = self._env.step(action, render=render)
            
        if done:
            time_step=ExtendedTimeStep(observation=ob,
                                step_type=StepType.LAST,
                                action=action,
                                reward=reward,
                                discount=1.0,
                                info=info)
        else:
            time_step=ExtendedTimeStep(observation=ob,
                                step_type=StepType.MID,
                                action=action,
                                reward=reward,
                                discount=1.0,
                                info=info)
            
        return self._augment_time_step(time_step)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                info=time_step.info or {})

    def observation_spec(self):
        return self._env.observation_spec

    def action_spec(self):
        return self._env.action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)
        

class HalfCheetahEnv(MujocoTrait):
    def __init__(self,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 task='default',
                 target_velocity=None,
                 model_path=None,
                 fixed_initial_state=False,
                 render_hw=100,
                 obs_type='states',
                 ):
        self._task = task
        self.obs_type = obs_type
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._task = task
        self._target_velocity = target_velocity
        self.fixed_initial_state = fixed_initial_state
        self.render_hw = render_hw
        
        self.half_cheetah_v4 = 1
        self._step_count = 0
        
        self._env=gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False, render_mode="rgb_array")

        if self.obs_type=='states':
            self.observation_spec = specs.BoundedArray(shape=self._env.observation_space.shape,
                                                            dtype='float32',
                                                            minimum=self._env.observation_space.low,
                                                            maximum=self._env.observation_space.high,
                                                            name='observation')
        elif self.obs_type=='pixels':
            self.observation_spec = specs.BoundedArray(shape=(3,96,96),
                                                            dtype='uint8',
                                                            minimum=0,
                                                            maximum=255,
                                                            name='observation')

        self.action_spec = specs.BoundedArray(shape=self._env.action_space.shape,
                                                          dtype='float32',
                                                          minimum=self._env.action_space.low,
                                                          maximum=self._env.action_space.high,
                                                          name='action')
        
    def get_obs(self):
        position = self._env.data.qpos.flat.copy()
        velocity = self._env.data.qvel.flat.copy()

        position = position[1:]

        return np.concatenate((position, velocity))
        
    def render(self):
        obs = self._env.render()
        return obs
        
    def step(self, action, render=False):
        if hasattr(self, '_step_count'):
            self._step_count += 1

        obsbefore = self.get_obs()
        xposbefore = self._env.data.qpos[0]
        observation, reward, terminated, truncated, info = self._env.step(action)
        obsafter = self.get_obs()
        xposafter = self._env.data.qpos[0]
        xvelafter = self._env.data.qvel[0]
        reward_ctrl = -0.1 * np.square(action).sum()
        
        reward = None
        if reward is None:
            if self._task == 'default':
                reward_vel = 0.
                reward_run = (xposafter - xposbefore) / self._env.dt
                reward = reward_ctrl + reward_run
            elif self._task == 'target_velocity':
                reward_vel = -(self._target_velocity - xvelafter) ** 2
                reward = reward_ctrl + reward_vel
            elif self._task == 'run_back':
                reward_vel = 0.
                reward_run = (xposbefore - xposafter) / self._env.dt
                reward = reward_ctrl + reward_run

        if self._step_count == 200:
            done = True
        else:
            done = False
        
        info["coordinates"] = np.array([xposbefore, 0.])
        info["next_coordinates"] = np.array([xposafter, 0.])
        info["ori_obs"] = obsbefore
        info["next_ori_obs"] = obsafter

        observation = observation.astype('float32')
        
        if self.obs_type == 'pixels':
            ob = self.render().transpose(2, 0, 1)   
            ob = ob.copy()
            
        if render:
            info['render'] = self._env.render().transpose(2, 0, 1)

        # done = terminated or truncated

        return observation, reward, done, info

    def reset(self):
        self._step_count=0
        observation, info = self._env.reset(seed=42)
        observation = observation.astype('float32')
        if self.obs_type == 'pixels':
            observation = self._env.render().transpose(2, 0, 1)   
            observation = observation.copy()
        return observation
        
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def plot_trajectory(self, trajectory, color, ax):
        # https://stackoverflow.com/a/20474765/2182622
        from matplotlib.collections import LineCollection
        linewidths = np.linspace(0.2, 1.2, len(trajectory))
        points = np.reshape(trajectory, (-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths, color=color)
        ax.add_collection(lc)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = super()._get_coordinates_trajectories(
                trajectories)
        for i, traj in enumerate(coordinates_trajectories):
            traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 1.25
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        coord_dims = [0]
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories, coord_dims)
        return eval_metrics