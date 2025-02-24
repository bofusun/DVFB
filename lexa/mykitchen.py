import akro
import sys
sys.path.append('lexa_env')
from lexa_envs import KitchenEnv
import numpy as np
from collections import OrderedDict, deque
from typing import Any, NamedTuple
import dm_env
from dm_env import StepType, specs

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

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
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
class MyKitchenEnv(KitchenEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_state = None
        self.last_ob = None
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.ob_info = dict(
            type='pixel',
            pixel_shape=(3, 84, 84),
        )

        # self._obs_spec = OrderedDict()
        # self._action_spec = OrderedDict()
        self._obs_spec = specs.BoundedArray(shape=self.observation_space.shape,
                                                          dtype='uint8',
                                                          minimum=0,
                                                          maximum=255,
                                                          name='observation')
        self._action_spec = specs.BoundedArray(shape=self.action_space.shape,
                                                          dtype=self.action_space.low.dtype,
                                                          minimum=self.action_space.low,
                                                          maximum=self.action_space.high,
                                                          name='action')
        
    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec
    
    @property
    def observation_space(self):
        return akro.Box(low=-np.inf, high=np.inf, shape=(3, 84, 84))

    def get_state(self, state):
        image = state['image']
        return image.flatten()

    def reset(self):
        state = super().reset()
        # ob = self.get_state(state)
        ob = state['image'].transpose(2, 0, 1).copy()
        self.last_state = state
        self.last_ob = ob
        time_step=ExtendedTimeStep(observation=ob,
                                step_type=StepType.FIRST,
                                action=np.zeros(self.action_space.shape, dtype=np.float32),
                                reward=0.0,
                                discount=1.0)
        return time_step

    def step(self, action, render=False):
        next_state, reward, done, info = super().step(action)
        # ob = self.get_state(next_state)
        ob = next_state['image'].transpose(2, 0, 1).copy()

        coords = self.last_state['state'][:2].copy()
        next_coords = next_state['state'][:2].copy()
        info['coordinates'] = coords
        info['next_coordinates'] = next_coords
        info['ori_obs'] = self.last_state['state']
        info['next_ori_obs'] = next_state['state']
        if render:
            info['render'] = next_state['image'].transpose(2, 0, 1)

        self.last_state = next_state
        self.last_ob = ob
        
        if done:
            time_step=ExtendedTimeStep(observation=ob,
                                step_type=StepType.LAST,
                                action=action,
                                reward=reward,
                                discount=1.0)
        else:
            time_step=ExtendedTimeStep(observation=ob,
                                step_type=StepType.MID,
                                action=action,
                                reward=reward,
                                discount=1.0)
        # return ob, reward, done, info
        return time_step

    def plot_trajectory(self, trajectory, color, ax):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        if plot_axis == 'free':
            return

        if plot_axis is None:
            plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]

        if plot_axis is not None:
            ax.axis(plot_axis)
            ax.set_aspect('equal')
        else:
            ax.axis('scaled')

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
        self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = []
        for trajectory in trajectories:
            if trajectory['env_infos']['coordinates'].dtype == object:
                coordinates_trajectories.append(np.concatenate([
                    np.concatenate(trajectory['env_infos']['coordinates'], axis=0),
                    [trajectory['env_infos']['next_coordinates'][-1][-1]],
                ]))
            elif trajectory['env_infos']['coordinates'].ndim == 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'],
                    [trajectory['env_infos']['next_coordinates'][-1]]
                ]))
            elif trajectory['env_infos']['coordinates'].ndim > 2:
                coordinates_trajectories.append(np.concatenate([
                    trajectory['env_infos']['coordinates'].reshape(-1, 2),
                    trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
                ]))
            else:
                assert False
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories, coord_dims=None):
        eval_metrics = {}

        goal_names = ['BottomBurner', 'LightSwitch', 'SlideCabinet', 'HingeCabinet', 'Microwave', 'Kettle']

        sum_successes = 0
        for i, goal_name in enumerate(goal_names):
            success = 0
            for traj in trajectories:
                success = max(success, traj['env_infos'][f'metric_success_task_relevant/goal_{i}'].max())
            eval_metrics[f'KitchenTask{goal_name}'] = success
            sum_successes += success
        eval_metrics[f'KitchenOverall'] = sum_successes

        return eval_metrics
