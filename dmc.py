from collections import OrderedDict, deque
from typing import Any, NamedTuple
import dataclasses
import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
import pickle
import gym
import custom_dmc_tasks as cdmc
import typing as tp

S = tp.TypeVar("S", bound="TimeStep")

@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self):
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self):
        return self.step_type == StepType.MID  # type: ignore

    def last(self):
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str):
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any):
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self
    
class EnvWrapper:
    def __init__(self, env):
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)
    
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


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((np.int(np.prod(spec.shape))
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action, render=False):
        time_step = self._env.step(action)
        info = {}
        if render:
            render_obs = self.physics.render(height=96,width=96, camera_id=0)
            info['render'] = render_obs.transpose(2, 0, 1).copy()  
        return self._augment_time_step(time_step, info, action)

    def _augment_time_step(self, time_step, info={}, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                info=info)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class Trajectory_Wrapper(dm_env.Environment):
    def __init__(self, env, domain):
        self._env = env
        self._domain = domain
        self._frame_skip = 1
        if domain == 'quadruped':
            self._camera_id = 2
        else:
            self._camera_id = 0
            
    def reset(self):
        time_step = self._env.reset()
        return time_step
    
    def step(self, action, render=False):
        time_step = self._env.step(action)
        info = {'internal_state': self._env.physics.get_state().copy()}
        xyz_before = self._env.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()
        obsbefore = self._env.physics.get_state()
        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
        xyz_after = self._env.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()
        obsafter = self._env.physics.get_state()
        
        if render:
            render_obs = self._env.physics.render(height=96,width=96, camera_id=self._camera_id)
            info['render'] = render_obs.transpose(2, 0, 1).copy()  

        if self._domain in ['cheetah', 'hopper']:
            info['coordinates'] = np.array([xyz_before[0], 0.])
            info['next_coordinates'] = np.array([xyz_after[0], 0.])
        elif self._domain in ['quadruped', 'humanoid']:
            info['coordinates'] = np.array([xyz_before[0], xyz_before[1]])
            info['next_coordinates'] = np.array([xyz_after[0], xyz_after[1]])
        info['ori_obs'] = obsbefore
        info['next_ori_obs'] = obsafter
        
        return self._augment_time_step(time_step, info, action)

    def _augment_time_step(self, time_step, info={}, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                info=info)
       
    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
     
    def plot_trajectory(self, trajectory, color, ax):
        if self._domain in ['walker', 'cheetah', 'hopper']:
            trajectory = trajectory.copy()
            # https://stackoverflow.com/a/20474765/2182622
            from matplotlib.collections import LineCollection
            linewidths = np.linspace(0.2, 1.2, len(trajectory))
            points = np.reshape(trajectory, (-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, linewidths=linewidths, color=color)
            ax.add_collection(lc)
        else:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        """Plot trajectories onto given ax."""
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
            coordinates_trajectories.append(np.concatenate([
                trajectory['env_infos']['coordinates'],
                [trajectory['env_infos']['next_coordinates'][-1]]
            ]))
        print("self._domain", self._domain)
        if self._domain in ['walker', 'cheetah', 'hopper']:
            for i, traj in enumerate(coordinates_trajectories):
                traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 1.25
                coordinates_trajectories[i] = traj
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = {}

        coord_dim = 2 if self._domain in ['quadruped', 'humanoid'] else 1

        coords = []
        for traj in trajectories:
            traj1 = traj['env_infos']['coordinates'][:, :coord_dim]
            traj2 = traj['env_infos']['next_coordinates'][-1:, :coord_dim]
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


class SparseMetaWorldStates:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
    ):
        import metaworld
        import os
        

        os.environ["MUJOCO_GL"] = "egl"

        # Construct the benchmark, sampling tasks
        self.ml1 = metaworld.ML1(f'{name}-v2', seed=seed) 

        # Create an environment with task `pick_place`
        env_cls = self.ml1.train_classes[f'{name}-v2']  
        self._env = env_cls()
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self._seed = seed
        self._tasks = self.ml1.test_tasks
        if name == 'reach':
            with open(f'/data/sjb_workspace/unsupervise_rl/choreographer-main/mw_tasks/reach_harder/{seed%10}.pickle', 'rb') as handle:
                self._tasks = pickle.load(handle)

    def observation_spec(self,):
        v = self.obs_space['observation']
        return specs.BoundedArray(name='observation', shape=v.shape, dtype=v.dtype, minimum=v.low, maximum=v.high)

    def action_spec(self,):
        return specs.BoundedArray(name='action',
            shape=self._env.action_space.shape, dtype=self._env.action_space.dtype, minimum=self._env.action_space.low, maximum=self._env.action_space.high)

    @property
    def obs_space(self):
        spaces = {
            "observation": self._env.observation_space,
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            state = state.astype('float32')
            success += float(info["success"])
            reward += float(rew)
        info["coordinates"] = self._state[:3] - self.init_coordinate
        info["obj_coordinates"] = self._state[4:7] - self.init_obj_coordinate
        self._state = state.copy()
        info["next_coordinates"] = state[:3] - self.init_coordinate
        info["next_obj_coordinates"] = state[4:7] - self.init_obj_coordinate
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        render_obs = self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy()
        info["render"] = render_obs
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "observation": state,
            "state": state,
            'action' : action,
            "success": success,
            'discount' : 1,
            'info':info
        }
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        # Set task to ML1 choices
        task_id = np.random.randint(0,len(self._tasks))
        return self.reset_with_task_id(task_id)

    def reset_with_task_id(self, task_id):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        
        # Set task to ML1 choices
        task = self._tasks[task_id]
        self._env.set_task(task)
        
        state = self._env.reset()
        self.init_coordinate = state[:3]
        self.init_obj_coordinate = state[4:7]
        self._state = state.copy()
        state = state.astype('float32')
        # This ensures the first observation is correct in the renderer
        self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera)
        for site in self._env._target_site_config:
            self._env._set_pos_site(*site)
        self._env.sim._render_context_offscreen._set_mujoco_buffers()

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": state,
            "state": state,
            'action' : np.zeros_like(self.act_space['action'].sample()),
            "success": False,
            'discount' : 1
        }
        return obs

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
        
    def __getattr__(self, name):
        if name == 'obs_space':
            return self.obs_space
        if name == 'act_space':
            return self.act_space
        return getattr(self._env, name)


class SparseMetaWorld:
    def __init__(
        self,
        name,
        seed=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
    ):
        import metaworld
        import os
        

        os.environ["MUJOCO_GL"] = "egl"

        # Construct the benchmark, sampling tasks
        self.ml1 = metaworld.ML1(f'{name}-v2', seed=seed) 

        # Create an environment with task `pick_place`
        env_cls = self.ml1.train_classes[f'{name}-v2']  
        self._env = env_cls()
        self._env._freeze_rand_vec = False
        self._size = size
        self._action_repeat = action_repeat
        self._camera = camera
        self._seed = seed
        self._tasks = self.ml1.test_tasks
        self._obs = []
        if name == 'reach':
            with open(f'/root/data1/sjb/unsupervise_rl/choreographer-main/mw_tasks/reach_harder/{seed}.pickle', 'rb') as handle:
                self._tasks = pickle.load(handle)

    def observation_spec(self,):
        v = self.obs_space['observation']
        return specs.BoundedArray(name='observation', shape=v.shape, dtype=v.dtype, minimum=v.low, maximum=v.high)

    def action_spec(self,):
        return specs.BoundedArray(name='action',
            shape=self._env.action_space.shape, dtype=self._env.action_space.dtype, minimum=self._env.action_space.low, maximum=self._env.action_space.high)

    @property
    def obs_space(self):
        spaces = {
            "observation": gym.spaces.Box(0, 255, (3,) + self._size, dtype=np.uint8),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": self._env.observation_space,
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
        }
        return spaces

    @property
    def act_space(self):
        action = self._env.action_space
        return {"action": action}

    def step(self, action):
        reward = 0.0
        success = 0.0
        for _ in range(self._action_repeat):
            state, rew, done, info = self._env.step(action)
            success += float(info["success"])
            reward += float(rew)
        info["coordinates"] = self._state[:3] - self.init_coordinate
        info["obj_coordinates"] = self._state[4:7] - self.init_obj_coordinate
        self._state = state.copy()
        info["next_coordinates"] = state[:3] - self.init_coordinate
        info["next_obj_coordinates"] = state[4:7] - self.init_obj_coordinate
        success = min(success, 1.0)
        assert success in [0.0, 1.0]
        render_obs = self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy()
        info["render"] = render_obs
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": False,  # will be handled by timelimit wrapper
            "is_terminal": False,  # will be handled by per_episode function
            "observation": render_obs,
            "state": state,
            'action' : action,
            "success": success,
            'discount' : 1,
            'info':info
        }
        return obs

    def reset(self):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        # Set task to ML1 choices
        task_id = np.random.randint(0,len(self._tasks))
        return self.reset_with_task_id(task_id)

    def reset_with_task_id(self, task_id):
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        
        # Set task to ML1 choices
        task = self._tasks[task_id]
        self._env.set_task(task)
        
        state = self._env.reset()
        self.init_coordinate = state[:3]
        self.init_obj_coordinate = state[4:7]
        self._state = state.copy()
        # This ensures the first observation is correct in the renderer
        self._env.sim.render(*self._size, mode="offscreen", camera_name=self._camera)
        for site in self._env._target_site_config:
            self._env._set_pos_site(*site)
        self._env.sim._render_context_offscreen._set_mujoco_buffers()

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "observation": self._env.sim.render(
                *self._size, mode="offscreen", camera_name=self._camera
            ).transpose(2, 0, 1).copy(),
            "state": state,
            'action' : np.zeros_like(self.act_space['action'].sample()),
            "success": False,
            'discount' : 1
        }
        return obs


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
        
    def __getattr__(self, name):
        if name == 'obs_space':
            return self.obs_space
        if name == 'act_space':
            return self.act_space
        return getattr(self._env, name)
    
class ExtendedTimeStepWrapperPlus(dm_env.Environment):
    def __init__(self, env, domain):
        self._env = env
        self.domain = domain
        if domain == 'quadruped':
            self._camera_id = 2
        else:
            self._camera_id = 0
            
    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action, render=False):
        if self.domain not in ["point_mass_maze"]:
            xyz_before = self.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()
        else:
            xyz_before = self.physics.position()
        time_step = self._env.step(action)
        if self.domain not in ["point_mass_maze"]:
            xyz_after = self.physics.named.data.geom_xpos[['torso'], ['x', 'y', 'z']].copy()
        else:
            xyz_after = self.physics.position()
        info = {}
        if render:
            render_obs = self.physics.render(height=96,width=96, camera_id=self._camera_id)
            info['render'] = render_obs.transpose(2, 0, 1).copy()  
        if self.domain in ['cheetah','walker', 'hopper']:
            info['coordinates'] = np.array([xyz_before[0], 0.])
            info['next_coordinates'] = np.array([xyz_after[0], 0.])
        elif self.domain in ['quadruped', 'humanoid', 'point_mass_maze']:
            info['coordinates'] = np.array([xyz_before[0], xyz_before[1]])
            info['next_coordinates'] = np.array([xyz_after[0], xyz_after[1]])
        return self._augment_time_step(time_step, info, action)

    def _augment_time_step(self, time_step, info={}, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                info=info)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def plot_trajectory(self, trajectory, color, ax):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2.0)

    def plot_trajectories(self, trajectories, colors, plot_axis, ax):
        square_axis_limit = 0.0
        for trajectory, color in zip(trajectories, colors):
            trajectory = np.array(trajectory)
            self.plot_trajectory(trajectory, color, ax)

            square_axis_limit = max(square_axis_limit, np.max(np.abs(trajectory[:, :2])))
        square_axis_limit = square_axis_limit * 1.2

        # if plot_axis == 'free':
        #     return

        # if plot_axis is None:
        #     plot_axis = [-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit]
        
         # 直接设置坐标轴范围
        if plot_axis is not None:
            ax.set_xlim(plot_axis[0], plot_axis[1])  # 设置 x 轴范围
            ax.set_ylim(plot_axis[2], plot_axis[3])  # 设置 y 轴范围
        else:
            ax.axis([-square_axis_limit, square_axis_limit, -square_axis_limit, square_axis_limit])
        
        ax.set_aspect('auto')  # 设置坐标轴比例为自动，允许自适应
        
        # 重新设置坐标轴范围，以确保不被覆盖
        ax.set_xlim(plot_axis[0], plot_axis[1])
        ax.set_ylim(plot_axis[2], plot_axis[3])

        # if plot_axis is not None:
        #     ax.axis(plot_axis)
        #     ax.set_aspect('equal')
        # else:
        #     ax.axis('scaled')
            
        # Add axis labels
        ax.set_xlabel('Position', fontsize=18)  # Replace with your desired label
        ax.set_ylabel('Skill Index', fontsize=18)  # Replace with your desired label
        
        # 美化：添加网格，设置线宽，调整颜色等
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_facecolor('white')  # 白色背景

        # 调整坐标轴刻度的字体大小
        ax.tick_params(axis='both', labelsize=18)  # 设置刻度字体大小

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
        if self.domain in ['walker', 'cheetah', 'hopper']:
            for i, traj in enumerate(coordinates_trajectories):
                traj[:, 1] = (i)
                coordinates_trajectories[i] = traj
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

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)
    # try:
    #   print("getattr(self._env, name)", getattr(self._env, name))
    #   return getattr(self._env, name)
    # except AttributeError:
    #   raise ValueError(name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
      time_step=ExtendedTimeStep(observation=obs['observation'],
                                step_type=StepType.LAST,
                                action=obs['action'],
                                reward=obs['reward'],
                                discount=1.0,
                                info=obs)
    else:
        time_step=ExtendedTimeStep(observation=obs['observation'],
                                step_type=StepType.MID,
                                action=obs['action'],
                                reward=obs['reward'],
                                discount=1.0,
                                info=obs)
    return time_step

  def reset(self):
    self._step = 0
    obs = self._env.reset()
    time_step = ExtendedTimeStep(observation=obs['observation'],
                                    step_type=StepType.FIRST,
                                    action=obs['action'],
                                    reward=obs['reward'],
                                    discount=1.0,
                                    info=obs)
    return time_step

  def reset_with_task_id(self, task_id):
    self._step = 0
    return self._env.reset_with_task_id(task_id)


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed):
    env = cdmc.make_jaco(task, obs_type, seed)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed):
    visualize_reward = False
    
    # env = cdmc.make(domain,
    #                         task,
    #                         task_kwargs=dict(random=seed),
    #                         environment_kwargs=dict(flat_observation=True),
    #                         visualize_reward=visualize_reward)
    
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    return env


def make(name, obs_type, frame_stack, action_repeat, seed):
    assert obs_type in ['states', 'pixels']
    if name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = name.split('_', 3)
    else:
        domain, task = name.split('_', 1)
    if domain == 'mw':
        if obs_type == 'pixels':
            return TimeLimit(SparseMetaWorld(task, seed=seed, action_repeat=action_repeat, size=(96,96), camera='corner2'), 250)
        else:
            return TimeLimit(SparseMetaWorldStates(task, seed=seed, action_repeat=action_repeat, size=(96,96), camera='corner2'), 250)
    elif domain == 'mw1':
        return TimeLimit(SparseMetaWorld(task, seed=seed, action_repeat=action_repeat, size=(96,96), camera='behindGripper'), 250)
    else:
        domain = dict(cup='ball_in_cup').get(domain, domain)

        make_fn = _make_jaco if domain == 'jaco' else _make_dmc
        env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed)

        if obs_type == 'pixels':
            env = FrameStackWrapper(env, frame_stack)
        else:
            env = ObservationDTypeWrapper(env, np.float32)

        env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
        env = ExtendedTimeStepWrapper(env)
        if domain not in 'jaco':
            env = ExtendedTimeStepWrapperPlus(env, domain=domain)
        return env
