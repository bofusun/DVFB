import akro
import gym
import numpy as np

from collections import defaultdict
import math
import os
import cv2
import dm_env
from gym import utils
from dm_env import StepType, specs
from typing import Any, NamedTuple
from gym.envs.mujoco import mujoco_env
from mujoco_envs.mujoco_utils import MujocoTrait

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
                                action=np.zeros(self.action_space.shape, dtype='float32'),
                                reward=0.0,
                                discount=1.0,
                                info={})
        return self._augment_time_step(time_step)

    def step(self, action):
        ob, reward, done, info = self._env.step(action)
            
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
        
class MazeEnv(gym.Env):
    def __init__(self, max_path_length, action_range=0.2):
        self.max_path_length = max_path_length
        self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = akro.Box(low=-action_range, high=action_range, shape=(2,))

        self.observation_spec = specs.BoundedArray(shape=self.observation_space.shape,
                                                            dtype='float32',
                                                            minimum=self.observation_space.low,
                                                            maximum=self.observation_space.high,
                                                            name='observation')

        self.action_spec = specs.BoundedArray(shape=self.action_space.shape,
                                                          dtype='float32',
                                                          minimum=self.action_space.low,
                                                          maximum=self.action_space.high,
                                                          name='action')
        
    def reset(self):
        self._cur_step = 0
        self._state = np.zeros(2)
        obs = self._state.astype('float32')
        return obs

    def step(self, action):
        obsbefore = self._state
        self._cur_step += 1
        self._state = self._state + action
        obs = self._state.astype('float32')
        obsafter = self._state
        done = self._cur_step >= self.max_path_length
        reward = obsafter[0] - obsbefore[0]
        return obs, reward, done, {
            'coordinates': obsbefore,
            'next_coordinates': obsafter,
            'ori_obs': obsbefore,
            'next_ori_obs': obsafter,
        }

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        rmin, rmax = None, None
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

            if rmin is None or rmin > np.min(trajectory[:, :2]):
                rmin = np.min(trajectory[:, :2])
            if rmax is None or rmax < np.max(trajectory[:, :2]):
                rmax = np.max(trajectory[:, :2])

        if plot_axis == 'nowalls':
            rcenter = (rmax + rmin) / 2.0
            rmax = rcenter + (rmax - rcenter) * 1.2
            rmin = rcenter + (rmin - rcenter) * 1.2
            plot_axis = [rmin, rmax, rmin, rmax]

        if plot_axis is not None:
            ax.axis(plot_axis)
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))

        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return {}
