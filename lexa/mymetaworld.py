import cv2
import gym
import sys
sys.path.append('/data/sjb_workspace/unsupervise_rl/url_benchmark-main/lexa_env/Metaworld')
import os
import numpy as np
import metaworld
import random
from collections import OrderedDict, deque
from typing import Any, NamedTuple
import dm_env
from dm_env import StepType, specs
import matplotlib.pyplot as plt
import metaworld.envs.mujoco.env_dict as _env_dict

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
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

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
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
        
    
class MyMetaWorldEnv(gym.Env):
    def __init__(self, env_name, obs_type, seed=0):
        self.env_name = env_name
        self.obs_type=obs_type
        self.seed=seed
        self.horizon=250
        self.meta_world=True
        
        Env = _env_dict.ALL_V2_ENVIRONMENTS[env_name] # get the class of the env
        self._env = Env(render_mode='rgb_array')
        self._env._freeze_rand_vec = False # when _freeze_rand_vec is set to True, calling reset() on the env will randomize the initial state
        self._env._set_task_called = True


        # self.all_v2 = metaworld.ALL_V2(env_name)
        # self._env = self.all_v2.train_classes[env_name](render_mode='rgb_array')
        # task = random.choice(self.all_v2.train_tasks)
        # self._env.set_task(task)
        # self._env.seed(self.seed)
        # print("self._env.observation_space", self._env.observation_space)
        # print("self._env.observation_space.shape", self._env.observation_space.shape)
        # print("self._env.observation_space.low", self._env.observation_space.low)
        if self.obs_type=='states':
            self._obs_spec = specs.BoundedArray(shape=self._env.observation_space.shape,
                                                            dtype='float32',
                                                            minimum=self._env.observation_space.low,
                                                            maximum=self._env.observation_space.high,
                                                            name='observation')
        elif self.obs_type=='pixels':
            self._obs_spec = specs.BoundedArray(shape=(3,96,96),
                                                            dtype='uint8',
                                                            minimum=0,
                                                            maximum=255,
                                                            name='observation')
        # print("self._env.action_space", self._env.action_space)
        # print("self._env.action_space.shape", self._env.action_space.shape)
        self._action_spec = specs.BoundedArray(shape=self._env.action_space.shape,
                                                          dtype='float32',
                                                          minimum=self._env.action_space.low,
                                                          maximum=self._env.action_space.high,
                                                          name='action')
        low, high = self._env.observation_space.low, self._env.observation_space.high
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._max_episode_steps = min(self.horizon, self._env.max_path_length)
        self._obs = []
    
    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec
    
    def reset(self):
        obs = self._env.reset()
        obs = obs[0].astype('float32')
        self.init_coordinate = obs[:3]
        self.init_obj_coordinate = obs[4:7]
        # print('init_pos', self._env.hand_init_pos)
        # print('init_coordinate', self.init_coordinate)
        self._obs = obs.copy()
        # print("reset obs0", obs)
        if self.obs_type == 'pixels':
            obs = self._env.render()
            # obs= obs.transpose(2, 0, 1).copy()
            obs = cv2.resize(obs, (96, 96)).transpose(2, 0, 1).copy()  
            # 翻转
            obs = np.flip(obs, axis=2).copy()  
        # print("reset obs", obs)
        time_step=ExtendedTimeStep(observation=obs,
                                step_type=StepType.FIRST,
                                action=np.zeros(self.action_space.shape, dtype='float32'),
                                reward=0.0,
                                discount=1.0,
                                info={})
        return time_step    
    
    
    def step(self, action, render=False):
        obs, reward, done, truncate, info = self._env.step(action)
        obs = obs.astype('float32')
        info["coordinates"] = self._obs[:3] - self.init_coordinate
        info["obj_coordinates"] = self._obs[4:7] - self.init_obj_coordinate
        # print("coordinates", info["coordinates"])
        # print("coordinates", self._obs[:3])
        # print("info_coordinates", info["coordinates"])
        self._obs = obs.copy()
        info["next_coordinates"] = obs[:3] - self.init_coordinate
        info["next_obj_coordinates"] = obs[4:7] - self.init_obj_coordinate
        # print("next_coordinates", info["next_coordinates"])
        # print("next_coordinates", self._obs[:3])
        # print("info_next_coordinates", info["next_coordinates"])
        if self.obs_type == 'pixels':
            obs = self._env.render()
            obs = cv2.resize(obs, (96, 96))
            obs = np.flip(obs, axis=0)
            obs = obs.transpose(2, 0, 1).copy()     
            # obs = cv2.resize(obs, (96, 96)).transpose(2, 0, 1).copy()  
            # # 翻转
            # obs = np.flip(obs, axis=0).copy()  
        done = done or truncate
        if render:
            render_obs = self._env.render()
            render_obs = np.flip(render_obs, axis=0).copy()
            info['render'] = render_obs.transpose(2, 0, 1).copy()  
            # from PIL import Image
            # print("render_obs", info['render'].shape)
            # # info['render'] = np.flip(render_obs, axis=0).copy()
            # img = Image.fromarray(info['render'].transpose(1, 2, 0))
            # img.save('/root/deeplearningnew/sun/unsupervise_rl/url_benchmark-main/lexa/render_obs.png')
        if done:
            time_step=ExtendedTimeStep(observation=obs,
                                step_type=StepType.LAST,
                                action=action,
                                reward=reward,
                                discount=1.0,
                                info=info)
        else:
            time_step=ExtendedTimeStep(observation=obs,
                                step_type=StepType.MID,
                                action=action,
                                reward=reward,
                                discount=1.0,
                                info=info)
        return time_step    
    
    def plot_trajectory(self, trajectory, color, ax):
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            # print("trajectory", trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :3])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        low_limit = self._env.hand_low-self.init_coordinate
        high_limit = self._env.hand_high-self.init_coordinate
        ax.set_xlim(low_limit[0], high_limit[0])
        ax.set_ylim(low_limit[1], high_limit[1])
        ax.set_zlim(low_limit[2], high_limit[2])
        # plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]
        # if plot_axis is None:
        #     plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        # if plot_axis is not None:
        #     ax.axis(plot_axis)
        #     ax.set_aspect('auto')
        # else:
        #     ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        obj_coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].dtype == object:
                coordinates_trajectories.append(np.concatenate([
                    np.concatenate(trajectory['env_infos']['coordinates'], axis=0),
                    [trajectory['env_infos']['next_coordinates'][-1][-1]],
                ]))
                obj_coordinates_trajectories.append(np.concatenate([
                    np.concatenate(trajectory['env_infos']['obj_coordinates'], axis=0),
                    [trajectory['env_infos']['next_obj_coordinates'][-1][-1]],
                ]))
            elif trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
                obj_coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['obj_coordinates'],
                    [trajectory['env_infos']['next_obj_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))
                obj_coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['obj_coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_obj_coordinates'].reshape(-1, 2)[-1:]
                ]))
            else:
                assert False
        return coordinates_trajectories






