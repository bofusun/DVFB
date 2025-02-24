import cv2
import gym
import sys
sys.path.append('lexa_env/Metaworld')
import numpy as np
import metaworld
import random
from collections import OrderedDict, deque
from typing import Any, NamedTuple
import dm_env
from dm_env import StepType, specs
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
        self.meta_world1=True
        
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
        self._obs = obs.copy()
        if self.obs_type == 'pixels':
            obs = self._env.render()
            # obs= obs.transpose(2, 0, 1).copy()
            obs = cv2.resize(obs, (96, 96)).transpose(2, 0, 1).copy()  
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
        info["coordinates"] = self._obs[:3]
        info["obj_coordinates"] = self._obs[4:7]
        self._obs = obs.copy()
        info["next_coordinates"] = obs[:3]
        info["next_obj_coordinates"] = obs[4:7]
        if self.obs_type == 'pixels':
            obs = self._env.render() 
            obs = cv2.resize(obs, (96, 96)).transpose(2, 0, 1).copy()  
        done = done or truncate
        if render:
            info['render']  = self._env.render().transpose(2, 0, 1).copy()  
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
    







