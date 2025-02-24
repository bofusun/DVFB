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
import numpy as np
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
        

class HalfCheetahEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

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
        utils.EzPickle.__init__(**locals())

        self.obs_type = obs_type
        if model_path is None:
            model_path = 'half_cheetah.xml'

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._task = task
        self._target_velocity = target_velocity
        self.fixed_initial_state = fixed_initial_state
        self.render_hw = render_hw

        self._step_count = 0
        
        xml_path = "mujoco_envs/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))

        mujoco_env.MujocoEnv.__init__(
            self,
            model_path,
            5)

        if self.obs_type=='states':
            self.observation_spec = specs.BoundedArray(shape=self.observation_space.shape,
                                                            dtype='float32',
                                                            minimum=self.observation_space.low,
                                                            maximum=self.observation_space.high,
                                                            name='observation')
        elif self.obs_type=='pixels':
            self.observation_spec = specs.BoundedArray(shape=(3,96,96),
                                                            dtype='uint8',
                                                            minimum=0,
                                                            maximum=255,
                                                            name='observation')
        # print("self._env.action_space", self._env.action_space)
        # print("self._env.action_space.shape", self._env.action_space.shape)
        self.action_spec = specs.BoundedArray(shape=self.action_space.shape,
                                                          dtype='float32',
                                                          minimum=self.action_space.low,
                                                          maximum=self.action_space.high,
                                                          name='action')
        
        
    def compute_reward(self, **kwargs):
        return None

    def _get_done(self):
        if self._step_count == 200:
            return True
        else:
            return False

    def step(self, action, render=False):
        self._step_count += 1
        obsbefore = self._get_obs()
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        obsafter = self._get_obs()
        xposafter = self.sim.data.qpos[0]
        xvelafter = self.sim.data.qvel[0]
        reward_ctrl = -0.1 * np.square(action).sum()

        reward = self.compute_reward(xposbefore=xposbefore, xposafter=xposafter)
        if reward is None:
            if self._task == 'default':
                reward_vel = 0.
                reward_run = (xposafter - xposbefore) / self.dt
                reward = reward_ctrl + reward_run
            elif self._task == 'target_velocity':
                reward_vel = -(self._target_velocity - xvelafter) ** 2
                reward = reward_ctrl + reward_vel
            elif self._task == 'run_back':
                reward_vel = 0.
                reward_run = (xposbefore - xposafter) / self.dt
                reward = reward_ctrl + reward_run

        if self._step_count == 200:
            done = True
        else:
            done = False
            
        ob = self._get_obs()
        info = dict(
            coordinates=np.array([xposbefore, 0.]),
            next_coordinates=np.array([xposafter, 0.]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        ob = ob.astype('float32')
        
        if self.obs_type == 'pixels':
            ob = self.render(mode='rgb_array', width=self.render_hw, height=self.render_hw).transpose(2, 0, 1)   
            ob = ob.copy()
        
        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)

        return ob, reward, done, info

    def _get_obs(self):
        if self._expose_all_qpos:
            obs = np.concatenate(
                [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]

        return obs

    def reset_model(self):
        self._step_count = 0
        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(
                low=-.1, high=.1, size=self.sim.model.nq)
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * .1
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        obs = obs.astype('float32')
        # print("reset obs0", obs)
        if self.obs_type == 'pixels':
            obs = self.render(mode='rgb_array', width=self.render_hw, height=self.render_hw).transpose(2, 0, 1)   
            obs = obs.copy()
            # obs = cv2.resize(obs, (84, 84)).copy()  
        
        return obs

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
