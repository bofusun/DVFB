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
from gym.envs.mujoco import mujoco_env
from mujoco_envs.mujoco_utils import MujocoTrait
# from gym.envs.mujoco import MuJocoPyEnv

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
        

# pylint: disable=missing-docstring
class AntEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 task="motion",
                 goal=None,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 expose_body_coms=None,
                 expose_body_comvels=None,
                 expose_foot_sensors=False,
                 use_alt_path=False,
                 model_path=None,
                 fixed_initial_state=False,
                 done_allowing_step_unit=None,
                 original_env=False,
                 render_hw=100,
                 obs_type='states',
                 ):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'ant.xml'

        self._task = task
        self._goal = goal
        self.obs_type = obs_type
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._expose_foot_sensors = expose_foot_sensors
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.fixed_initial_state = fixed_initial_state

        self._done_allowing_step_unit = done_allowing_step_unit
        self._original_env = original_env
        self.render_hw = render_hw

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py

        self._step_count = 0
        
        self.normalizer_mean = np.array(
            [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
             0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
             0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
             -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855])
        self.normalizer_std = np.array(
            [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
             0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
             0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
             4.3704205, 4.5175962, 4.3704395])
        
        # self._obs_mean = normalizer_mean
        # self._obs_var  = normalizer_std ** 2

        self._obs_mean = np.full((29,), 0 if self.normalizer_mean is None else self.normalizer_mean)
        self._obs_var = np.full((29,), 1 if self.normalizer_std is None else self.normalizer_std ** 2)    
        
        # observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64
        # )
        # print("self.action_space.shape", self.action_space.shape)
        # print("self.observation_space.shape", self.observation_space.shape)
                
        xml_path = "mujoco_envs/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)
        # MuJocoPyEnv.__init__(self, model_path, 5)
        
        # print("self.observation_space.shape", self.observation_space.shape)
        # while True:
        #     pass
     
        # self._obs_mean = np.full(self.observation_space.shape, 0 if normalizer_mean is None else normalizer_mean)
        # self._obs_var = np.full(self.observation_space.shape, 1 if normalizer_std is None else normalizer_std ** 2)    
        # print("self.observation_space.shape", self.observation_space.shape)    
        
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
        # print("self.action_space.shape", self.action_space.shape)
        # print("self.observation_space.shape", self.observation_space.shape)
        
    def compute_reward(self, **kwargs):
        return None

    def _apply_normalize_obs(self, obs):
        normalized_obs = (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        return normalized_obs
    
    def _get_done(self):
        if self._step_count == 200:
            return True
        else:
            return False

    def step(self, a, render=False):
        # print("self.action_space.low", self.action_space.low)
        # print("self.action_space.high", self.action_space.high)
        if hasattr(self, '_step_count'):
            self._step_count += 1

        obsbefore = self._get_obs()
        xposbefore = self.sim.data.qpos.flat[0]
        yposbefore = self.sim.data.qpos.flat[1]
        self.do_simulation(a, self.frame_skip)
        obsafter = self._get_obs()
        xposafter = self.sim.data.qpos.flat[0]
        yposafter = self.sim.data.qpos.flat[1]

        reward = self.compute_reward(xposbefore=xposbefore, yposbefore=yposbefore, xposafter=xposafter, yposafter=yposafter)
        if reward is None:
            forward_reward = (xposafter - xposbefore) / self.dt
            sideward_reward = (yposafter - yposbefore) / self.dt

            ctrl_cost = .5 * np.square(a).sum()
            survive_reward = 1.0
            if self._task == "forward":
                reward = forward_reward - ctrl_cost + survive_reward
            elif self._task == "backward":
                reward = -forward_reward - ctrl_cost + survive_reward
            elif self._task == "left":
                reward = sideward_reward - ctrl_cost + survive_reward
            elif self._task == "right":
                reward = -sideward_reward - ctrl_cost + survive_reward
            elif self._task == "goal":
                reward = -np.linalg.norm(np.array([xposafter, yposafter]) - self._goal)
            elif self._task == "motion":
                reward = np.max(np.abs(np.array([forward_reward, sideward_reward
                                                 ]))) - ctrl_cost + survive_reward

            def _get_gym_ant_reward():
                forward_reward = (xposafter - xposbefore)/self.dt
                ctrl_cost = .5 * np.square(a).sum()
                contact_cost = 0.5 * 1e-3 * np.sum(
                    np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
                survive_reward = 1.0
                reward = forward_reward - ctrl_cost - contact_cost + survive_reward
                return reward
            reward = _get_gym_ant_reward()
            

        if self._step_count == 200:
            done = True
        else:
            done = False

        ob = self._get_obs()
        
        info = dict(
            # reward_forward=forward_reward,
            # reward_sideward=sideward_reward,
            # reward_ctrl=-ctrl_cost,
            # reward_survive=survive_reward,
            coordinates=np.array([xposbefore, yposbefore]),
            next_coordinates=np.array([xposafter, yposafter]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        # ob = self._apply_normalize_obs(ob)
        ob = ob.astype('float32')
        
        if self.obs_type == 'pixels':
            ob = self.render(mode='rgb_array', width=self.render_hw, height=self.render_hw).transpose(2, 0, 1)   
            ob = ob.copy()
            
        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)
            # print("info['render']", info['render'].shape) #[3,256,256]

        return ob, reward, done, info

    def _get_obs(self):
        if self._original_env:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])

        # No crfc observation
        if self._expose_all_qpos:
            obs = np.concatenate([
                self.sim.data.qpos.flat[:15],
                self.sim.data.qvel.flat[:14],
            ])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat[2:15],
                self.sim.data.qvel.flat[:14],
            ])

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])

        if self._expose_foot_sensors:
            obs = np.concatenate([obs, self.sim.data.sensordata])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]
            
        # obs = self._apply_normalize_obs(obs)

        return obs


    def reset_model(self):
        self._step_count = 0
        self._done_internally = False

        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(
                size=self.sim.model.nq, low=-.1, high=.1)
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * .1

        if not self._original_env:
            qpos[15:] = self.init_qpos[15:]
            qvel[14:] = 0.

        self.set_state(qpos, qvel)
        
        obs = self._get_obs()
        # obs = self._apply_normalize_obs(obs)
        obs = obs.astype('float32')
        # print("reset obs0", obs)
        if self.obs_type == 'pixels':
            obs = self.render(mode='rgb_array', width=self.render_hw, height=self.render_hw).transpose(2, 0, 1)   
            obs = obs.copy()
            # obs= obs.transpose(2, 0, 1).copy()     
            # obs = cv2.resize(obs, (96, 96)).copy()  
        
        return obs

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 2.5
        pass

    @property
    def body_com_indices(self):
        return self._body_com_indices

    @property
    def body_comvel_indices(self):
        return self._body_comvel_indices

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        coord_dims = [0, 1]
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories, coord_dims)
        return eval_metrics
