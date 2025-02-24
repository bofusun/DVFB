import cv2
import imageio
import numpy as np
import wandb


class VideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
                self.frames.append(frame)
            elif hasattr(env, '_camera'):
                frame = env._env.sim.render(*(self.render_size,self.render_size), camera_name=env._camera).copy()
                self.frames.append(frame)
            elif hasattr(env, 'use_goal_idx'):
                frame = env._env.render('rgb_array', width=self.render_size, height=self.render_size)
                self.frames.append(frame)
            elif hasattr(env, 'meta_world'):
                frame = env._env.render()
                # 翻转
                frame = np.flip(frame, axis=0)
                self.frames.append(frame)
            elif hasattr(env, 'meta_world1'):
                frame = env._env.render()
                self.frames.append(frame)
            elif hasattr(env, 'ant_v4'):
                frame = env.render()
                self.frames.append(frame)
            elif hasattr(env, 'half_cheetah_v4'):
                frame = env.render()
                self.frames.append(frame)
            elif hasattr(env, 'ant'):
                frame = env.render(mode='rgb_array', width=self.render_size, height=self.render_size)
                self.frames.append(frame)
            elif hasattr(env, '_expose_obs_idxs'):
                frame = env.render(mode='rgb_array', width=self.render_size, height=self.render_size)
                self.frames.append(frame)
            elif hasattr(env, 'max_path_length'):
                pass
            else:
                frame = env.render()
                self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'eval/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            # if self.use_wandb:
            #     self.log_to_wandb()
            if not self.frames == []:
                path = self.save_dir / file_name
                imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0,
                 use_wandb=False):
        if root_dir is not None:
            self.save_dir = root_dir
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
                self.frames.append(frame)
            elif hasattr(env, '_camera'):
                frame = env._env.sim.render(*(self.render_size,self.render_size), camera_name=env._camera).copy()
                self.frames.append(frame)
            elif hasattr(env, 'use_goal_idx'):
                frame = env._env.render('rgb_array', width=self.render_size, height=self.render_size)
                self.frames.append(frame)
            elif hasattr(env, 'meta_world'):
                frame = env._env.render()
                # 翻转
                frame = np.flip(frame, axis=0)
                self.frames.append(frame)
            elif hasattr(env, 'meta_world1'):
                frame = env._env.render()
                self.frames.append(frame)
            elif hasattr(env, 'ant_v4'):
                frame = env.render()
                self.frames.append(frame)
            elif hasattr(env, 'half_cheetah_v4'):
                frame = env.render()
                self.frames.append(frame)
            elif hasattr(env, 'ant'):
                frame = env.render(mode='rgb_array', width=self.render_size, height=self.render_size)
                self.frames.append(frame)
            elif hasattr(env, '_expose_obs_idxs'):
                frame = env.render(mode='rgb_array', width=self.render_size, height=self.render_size)
                self.frames.append(frame)
            elif hasattr(env, 'max_path_length'):
                pass
            else:
                frame = env.render()
                self.frames.append(frame)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log({
            'train/video':
            wandb.Video(frames[::skip, :, ::2, ::2], fps=fps, format="gif")
        })

    def save(self, file_name):
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
