from collections import OrderedDict

import akro
import numpy as np
from gym import spaces
import gym
import gym.spaces
import gym.spaces.utils

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = akro.Box(low=low, high=high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoTrait:
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = akro.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def render(self,
               mode='human',
               width=100,
               height=100,
               camera_id=None,
               camera_name=None):
        if hasattr(self, 'render_hw') and self.render_hw is not None:
            width = self.render_hw
            height = self.render_hw
        return super().render(mode=mode, width=width, height=height)

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

        if coord_dims is not None:
            coords = []
            for traj in trajectories:
                traj1 = traj['env_infos']['coordinates'][:, coord_dims]
                traj2 = traj['env_infos']['next_coordinates'][-1:, coord_dims]
                coords.append(traj1)
                coords.append(traj2)
            coords = np.concatenate(coords, axis=0)
            uniq_coords = np.unique(np.floor(coords), axis=0)
            eval_metrics.update({
                'MjNumTrajs': len(trajectories),
                'MjAvgTrajLen': len(coords) / len(trajectories) - 1,
                'MjNumCoords': len(coords),
                'MjNumUniqueCoords': len(uniq_coords),
            })

        return eval_metrics



# @dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""
    input_space: akro.Space
    output_space: akro.Space
    
class EnvSpec(InOutSpec):
    """Describes the action and observation spaces of an environment.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.

    """

    def __init__(self, observation_space, action_space):
        super().__init__(action_space, observation_space)

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space

    def __eq__(self, other):
        """See :meth:`object.__eq__`.

        Args:
            other (EnvSpec): :class:`~EnvSpec` to compare with.

        Returns:
            bool: Whether these :class:`~EnvSpec` instances are equal.

        """
        return (self.observation_space == other.observation_space
                and self.action_space == other.action_space)

class AkroWrapperTrait:
    @property
    def spec(self):
        return EnvSpec(action_space=akro.from_gym(self.action_space),
                       observation_space=akro.from_gym(self.observation_space))
        
class ConsistentNormalizedEnv(AkroWrapperTrait, gym.Wrapper):
    def __init__(
            self,
            env,
            expected_action_scale=1.,
            flatten_obs=True,
            normalize_obs=True,
            mean=None,
            std=None,
    ):
        super().__init__(env)

        self._normalize_obs = normalize_obs
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs
        self.ant = 1

        self._obs_mean = np.full(env.observation_space.shape, 0 if env.normalizer_mean is None else env.normalizer_mean)
        self._obs_var = np.full(env.observation_space.shape, 1 if env.normalizer_std is None else env.normalizer_std ** 2)

        self._cur_obs = None

        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_space = akro.Box(low=-self._expected_action_scale,
                                         high=self._expected_action_scale,
                                         shape=self.env.action_space.shape)
        else:
            self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _apply_normalize_obs(self, obs):
        normalized_obs = (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        return normalized_obs

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._cur_obs = obs
        if self._normalize_obs:
            obs = self._apply_normalize_obs(obs)

        if self._flatten_obs:
            obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        return obs

    def step(self, action, **kwargs):
        if isinstance(self.env.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            lb, ub = self.env.action_space.low, self.env.action_space.high
            # print("lb", lb)
            # print("ub", ub)
            if np.all(lb != -np.inf) and np.all(ub != -np.inf):
                scaled_action = lb + (action + self._expected_action_scale) * (
                        0.5 * (ub - lb) / self._expected_action_scale)
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action


    
        next_obs, reward, done, info = self.env.step(scaled_action, **kwargs)
        info['original_observations'] = self._cur_obs
        info['original_next_observations'] = next_obs

        self._cur_obs = next_obs

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)

        if self._flatten_obs:
            next_obs = gym.spaces.utils.flatten(self.env.observation_space, next_obs)

        return next_obs, reward, done, info


consistent_normalize = ConsistentNormalizedEnv
