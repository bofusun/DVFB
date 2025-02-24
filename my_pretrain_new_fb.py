import os
import dmc
import dmc1
import time
import copy
import utils
import hydra
import torch
import wandb
import pickle
import argparse
import warnings
import numpy as np
from matplotlib import figure
from utils import get_option_colors
from dm_env import specs
from pathlib import Path
from logger import Logger
from utils import record_video
from collections import OrderedDict
from dmc_benchmark import PRIMAL_TASKS
from video import TrainVideoRecorder, VideoRecorder
from replay_buffer_fb import ReplayBufferStorage, make_replay_loader
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
warnings.filterwarnings('ignore', category=DeprecationWarning)

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.domain=cfg.domain
        self.temp_dir = cfg.work_dir
        self.obs_type = cfg.obs_type
        self.seed = cfg.seed
        # discretes = {'aps':0, 'diayn':1, 'cic':0, 'metra':0, 'smm':1, 'smm1':0, 'smm2':0, 'ddpg':0, 'disagreement':0, 'icm_apt':0, \
        #              'icm':0, 'proto':0, 'rnd':0, 'smm3':0, 'smm4':0, 'smm5':0, 'smm6':0, 'smm7':0, 'smm8':0, 'smm9':0, 'smm10':0, \
        #              'smm11':0, 'smm12':0, 'smm13':0, 'smm14':0, 'smm20':0, 'smm21':0, 'smm22':0, 'smm23':0, 'smm24':0, 'smm25':0, \
        #              'smm26':0, 'smm27':0, 'smm28':0, 'smm29':0, 'smm30':0, 'smm37':0, 'smm38':0, 'smm39':0, 'smm40':0, 'smm41':0,\
        #              'smm42':0, 'smm43':0, 'smm44':0, 'smm45':0, 'smm46':0, 'smm52':0, 'smm53':0, 'smm54':0, 'smm55':0, 'smm56':0, \
        #              'smm57':0, 'smm58':0, 'smm59':0, 'smm60':0, 'smm61':0, 'smm62':0, 'smm63':0, 'smm64':0, 'smm65':0, 'smm66':0, \
        #              'smm67':0, 'smm68':0, 'smm69':0, 'smm75':0, 'smm76':0, 'smm77':0, 'smm78':0, 'smm79':0, 'smm80':0, 'smm97':0, \
        #              'smm98':0, 'smm99':0, 'smm100':0, 'smm101':0, 'smm102':0, 'smm103':0, 'smm104':0, 'smm105':0, 'smm115':0,\
        #              'smm116':0, 'smm117':0, 'smm118':0, 'smm119':0, 'smm120':0, 'smm121':0, 'smm122':0, 'smm123':0, 'smm124':0, \
        #              'smm172':0, 'smm202':0, 'smm203':0, 'smm204':0, 'smm205':0, 'smm206':0, 'smm230':0, 'smm231':0, 'smm232':0, \
        #              'smm233':0, 'smm207':0, 'smm208':0, 'smm209':0, 'smm210':0}
        # self.discrete= discretes[str(cfg.agent.name)]
        self.discrete=0
        if cfg.agent.name in ["aps"]:
            self.skill_dim = cfg.agent.sf_dim
        elif cfg.agent.name in ["smm","smm1","smm2","smm3","smm4","smm5","smm6"]:
            self.skill_dim = cfg.agent.z_dim
        elif cfg.agent.name in ["diayn", "cic", "metra","smm7","smm8","smm9","smm10","smm11","smm12","smm13","smm14","smm20","smm21", \
                                'smm22', 'smm23', 'smm24', 'smm25', 'smm26', 'smm27', 'smm28', 'smm29', 'smm30', 'smm37', "smm38", \
                                 "smm39", "smm40", "smm41", "smm42", "smm43", "smm44", "smm45", "smm46", "smm52", "smm53", "smm54",\
                                 "smm55", "smm56", "smm57", "smm58", "smm59", 'smm60', 'smm61', 'smm62', 'smm63', "smm64", "smm65", \
                                 "smm66", "smm67", "smm68", "smm69", "smm75", "smm76", "smm77", "smm78", "smm79", "smm80", "smm97", \
                                 "smm98", "smm99", "smm100", "smm101", "smm102", "smm103", "smm104", "smm105", "smm115", "smm116", \
                                 "smm117", "smm118", "smm119", "smm120", "smm121", "smm122", "smm123", "smm124", "smm125", "smm172", \
                                 "smm202", "smm203", "smm204", "smm205", "smm206", "smm230", "smm231", "smm232", "smm233", "smm207", \
                                 "smm208", "smm209", "smm210"]:
            self.skill_dim = cfg.agent.skill_dim
        else:
            self.skill_dim = cfg.agent.skill_dim
        # 保存文件位置
        self.work_dir = os.path.join(str(self.domain)+'_'+str(self.obs_type)+'_pretrain', str(cfg.agent.name), str(self.seed))
        print("self.work_dir", self.work_dir)
        utils.make_dir(self.work_dir)
        self.video_dir = utils.make_dir(os.path.join(self.work_dir, 'video'))
        self.model_dir = utils.make_dir(os.path.join(self.work_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.work_dir, 'buffer'))
        self.traj_dir = utils.make_dir(os.path.join(self.work_dir, 'trajectory'))
        # 种子与设备
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb_pretrain_"+str(cfg.domain)+str(cfg.obs_type), group=cfg.agent.name, name=exp_name)
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # 环境设置 
        self.train_env, self.eval_env = self.set_env()
        # 智能体
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        # 回放池
        self._replay_iter = None
        self.replay_storage, self.replay_loader = self.create_replay_buffer()

        # create video recorders
        self.video_recorder = VideoRecorder(
            Path(self.video_dir) if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            Path(self.video_dir) if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def set_env(self):
        if self.domain in ['walker', 'cheetah', 'quadruped', 'point_mass_maze', 'humanoid', 'hopper', 'jaco', 'mw', 'mw1']:
            task = PRIMAL_TASKS[self.domain]
            print("task", task)
            train_env = dmc.make(task, self.cfg.obs_type, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
            eval_env = dmc.make(task, self.cfg.obs_type, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
        elif self.domain in ['ant']:
            from mujoco_envs.ant_env_new import AntEnv, ExtendedTimeStepWrapper
            task = PRIMAL_TASKS[self.domain]
            train_env = ExtendedTimeStepWrapper(AntEnv(render_hw=96, obs_type = self.cfg.obs_type))
            eval_env = ExtendedTimeStepWrapper(AntEnv(render_hw=96, obs_type = self.cfg.obs_type))
        elif self.domain in ['half_cheetah']:
            from mujoco_envs.half_cheetah_env_new import HalfCheetahEnv, ExtendedTimeStepWrapper
            task = PRIMAL_TASKS[self.domain]
            train_env = ExtendedTimeStepWrapper(HalfCheetahEnv(render_hw=96, obs_type = self.cfg.obs_type))
            eval_env = ExtendedTimeStepWrapper(HalfCheetahEnv(render_hw=96, obs_type = self.cfg.obs_type))
        elif self.domain in ['maze']:
            from mujoco_envs.maze_env import MazeEnv, ExtendedTimeStepWrapper 
            train_env = ExtendedTimeStepWrapper(MazeEnv(max_path_length=200, action_range=0.2,))
            eval_env = ExtendedTimeStepWrapper(MazeEnv(max_path_length=200, action_range=0.2,))
        elif self.domain in ['kitchen']:
            from lexa.mykitchen_new import MyKitchenEnv
            train_env = MyKitchenEnv(log_per_goal=True)
            eval_env = MyKitchenEnv(log_per_goal=True)
        # elif self.domain in ['metaworld']:
        #     from lexa.mymetaworld import MyMetaWorldEnv
        #     task = 'pick-place-v2'
        #     train_env = MyMetaWorldEnv(task, self.cfg.obs_type, self.cfg.seed)
        #     eval_env = MyMetaWorldEnv(task, self.cfg.obs_type, self.cfg.seed+42)
        # elif self.domain in ['metaworld1']:
        #     from lexa.mymetaworld_firstview import MyMetaWorldEnv
        #     task = 'pick-place-v2'
        #     train_env = MyMetaWorldEnv(task, self.obs_type, self.cfg.seed)
        #     eval_env = MyMetaWorldEnv(task, self.obs_type, self.cfg.seed+42)
        return train_env, eval_env

    def create_replay_buffer(self):
        # 数据空间
        meta_specs = self.agent.get_meta_specs()
        data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'))
        # 创建数据空间
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs, Path(self.buffer_dir))
        # 创建回放池
        self.replay_loader = make_replay_loader(self.replay_storage, self.cfg.replay_buffer_size, self.cfg.batch_size, self.cfg.replay_buffer_num_workers, \
                                                False, self.cfg.nstep, self.cfg.discount)
        return self.replay_storage, self.replay_loader

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter
    
    def create_meta(self, meta_value):
        meta = OrderedDict()
        meta_specs = self.agent.get_meta_specs()
        # print("meta_specs", meta_specs)
        # print("meta_specs[-1]", meta_specs[0].name)
        if meta_specs != tuple():
            meta[meta_specs[0].name] = meta_value
        return meta
    
    # 评估轨迹
    def evaluate_diversity(self):      
        if self.discrete:
            eye_options = np.eye(self.skill_dim)
            random_options = []
            colors = []
            # 为每个option添加一些轨迹
            for i in range(self.skill_dim):
                num_trajs_per_option = self.cfg.num_random_trajectories // self.skill_dim + (i < self.cfg.num_random_trajectories % self.skill_dim)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            # random_options中为许多option，colors为对应的颜色
            random_options = np.array(random_options)
            random_options = random_options.astype('float32')
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.skill_dim <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
            options = []
            for option in random_options:
                option_meta = self.create_meta(option)
                options.append(option_meta)
        else:
            # print("self.param.skill_dim", self.param.skill_dim)
            # 采样一些随机的option,每个轨迹一个option [48, 2]
            random_options = np.random.randn(self.cfg.num_random_trajectories, self.skill_dim)
            if self.cfg.unit_meta:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
                random_options = random_options.astype('float32')
            random_option_colors = get_option_colors(random_options * 4)
            options = []
            for option in random_options:
                option_meta = self.agent.init_meta()
                options.append(option_meta)
        
        if self.domain in ['walker', 'ant','half_cheetah', 'maze', 'quadruped', 'humanoid', 'cheetah', 'hopper', 'point_mass_maze']:
            trajectories = self.get_trajectories(options)
            fig = figure.Figure()
            ax = fig.add_subplot()
            eval_plot_axis=[-50.0, 50.0, -50.0, 50.0]
            self.eval_env.render_trajectories(trajectories, random_option_colors, eval_plot_axis, ax)
            fig.savefig(os.path.join(self.traj_dir, 'diversity_'+str(self.global_step)+'.png'), dpi=300)
        elif self.domain in ['mw','mw1']:
            trajectories = self.get_3d_trajectories(options)
            fig = figure.Figure()
            ax = fig.add_subplot(projection='3d')
            eval_plot_axis=[-50.0, 50.0, -50.0, 50.0]
            self.eval_env.render_trajectories(trajectories, random_option_colors, eval_plot_axis, ax)
            fig.savefig(os.path.join(self.traj_dir, 'diversity_'+str(self.global_step)+'.png'), dpi=300)
        save_tajectories = {}
        save_tajectories['trajectories'] = trajectories
        save_tajectories['random_option_colors'] = random_option_colors
        with open(os.path.join(self.traj_dir, "trajectory_"+str(self.global_step)+".pkl"), "wb") as f:
            pickle.dump(save_tajectories, f, pickle.HIGHEST_PROTOCOL)   

    def evaluate_video(self):
        if self.discrete:
            video_options = np.eye(self.skill_dim)
            video_options = video_options.repeat(self.cfg.num_video_repeats, axis=0)
            video_options = video_options.astype('float32')
            options = []
            for option in video_options:
                option_meta = self.create_meta(option)
                options.append(option_meta)
        else:
            if self.skill_dim == 2:
                radius = 1. if self.cfg.unit_meta else 1.5
                video_options = []
                for angle in [3, 2, 1, 4]:
                    video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                video_options.append([0, 0])
                for angle in [0, 5, 6, 7]:
                    video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                video_options = np.array(video_options)
            else:
                video_options = np.random.randn(9, self.skill_dim)
                if self.cfg.unit_meta:
                    video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
            video_options = video_options.repeat(self.cfg.num_video_repeats, axis=0)
            video_options = video_options.astype('float32')
            options = []
            for option in video_options:
                option_meta = self.agent.init_meta()
                options.append(option_meta)

        video_trajectories = self.get_video_trajectories(options)
        plot_path = os.path.join(self.video_dir, 'traj_video_'+str(self.global_step)+'.mp4')
        record_video(plot_path, video_trajectories, skip_frames=self.cfg.video_skip_frames)        

    def get_video_trajectories(self, options):
        meta_specs = self.agent.get_meta_specs()
        trajectories = []
        for meta in options:
            trajectory = {}        
            render = []
            time_step = self.eval_env.reset()    
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                if self.domain in ['mw', 'mw1']:    
                    time_step = self.eval_env.step(action)       
                    render.append(time_step.info["info"]['render'])
                else:
                    time_step = self.eval_env.step(action, render=True)  
                    render.append(time_step.info['render'])
            trajectory['env_infos']={}
            trajectory['env_infos']['render']=np.array(render)
            trajectories.append(trajectory)
        return trajectories
    
    # 获取轨迹
    def get_trajectories(self, options):
        meta_specs = self.agent.get_meta_specs()
        trajectories = []
        for meta in options:
            trajectory = {}        
            observations = []
            actions = []
            rewards = []
            discount = []
            dones = []
            options = []
            coordinates = []
            next_coordinates = []
            time_step = self.eval_env.reset()    
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                time_step = self.eval_env.step(action)            
                observations.append(time_step.observation)
                actions.append(time_step.action)
                rewards.append(time_step.reward)
                discount.append(time_step.discount)
                dones.append(time_step.last())
                options.append(meta)
                coordinates.append(time_step.info['coordinates'])
                next_coordinates.append(time_step.info['next_coordinates'])
            trajectory['observations']=np.array(observations)
            trajectory['actions']=np.array(actions)
            trajectory['rewards']=np.array(rewards)
            trajectory['discount']=np.array(discount)
            trajectory['dones']=np.array(dones)
            trajectory['options']=np.array(options)
            trajectory['env_infos']={}
            trajectory['env_infos']['coordinates']=np.array(coordinates)
            trajectory['env_infos']['next_coordinates']=np.array(next_coordinates)
            trajectories.append(trajectory)
        return trajectories

    def get_3d_trajectories(self, options):
        meta_specs = self.agent.get_meta_specs()
        trajectories = []
        for meta in options:
            trajectory = {}        
            options = []
            coordinates = []
            next_coordinates = []
            obj_coordinates = []
            next_obj_coordinates = []
            time_step = self.eval_env.reset()    
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                time_step = self.eval_env.step(action)       
                if meta_specs==():
                    options.append([])
                else:      
                    options.append(meta[meta_specs[0].name])
                coordinates.append(time_step.info['info']['coordinates'])
                next_coordinates.append(time_step.info['info']['next_coordinates'])
                obj_coordinates.append(time_step.info['info']['obj_coordinates'])
                next_obj_coordinates.append(time_step.info['info']['next_obj_coordinates'])
            trajectory['options']=np.array(options)
            trajectory['env_infos']={}
            trajectory['env_infos']['coordinates']=np.array(coordinates)
            trajectory['env_infos']['next_coordinates']=np.array(next_coordinates)
            trajectory['env_infos']['obj_coordinates']=np.array(obj_coordinates)
            trajectory['env_infos']['next_obj_coordinates']=np.array(next_obj_coordinates)
            trajectories.append(trajectory)
        return trajectories

    def evaluate(self):
        success, step, episode, total_reward = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1
            if self.domain == 'metaworld' or self.domain == 'metaworld1':
                success += time_step.info['success']
            episode += 1
            self.video_recorder.save(f'eval_{self.global_frame}.mp4')
        success_ratio = success / self.cfg.num_eval_episodes
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('sucess_ratio', success_ratio)
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        if self.domain not in ['jaco']:
            self.evaluate_diversity()
        # if self.domain in ['ant', 'half_cheetah','maze', 'quadruped', 'humanoid', 'metaworld', 'metaworld1']:
        #     self.evaluate_diversity()
        self.evaluate_video()

    def evaluate_train(self):
        success, step, episode, total_reward = 0, 0, 0, 0
        eval_until_episode = utils.Until(1)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            step=0
            time_step = self.train_env.reset()
            self.train_video_recorder.init(self.train_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=True)
                time_step = self.train_env.step(action)
                self.train_video_recorder.record(self.train_env)
                total_reward += time_step.reward
                step += 1
            episode += 1
            self.train_video_recorder.save(f'train_{self.global_frame}.mp4')  
        with self.logger.log_and_dump_ctx(self.global_frame, ty='test') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        self.agent.solved_meta = None 

    def infer_meta(self):
        # 数据空间
        meta_specs = self.agent.get_meta_specs()
        data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'))
        # 创建数据空间
        replay_storage = ReplayBufferStorage(data_specs, meta_specs, Path(self.buffer_dir))
        # 创建回放池
        replay_loader = make_replay_loader(replay_storage, self.cfg.replay_buffer_size, self.cfg.batch_size, self.cfg.replay_buffer_num_workers, \
                                                False, self.cfg.nstep, self.cfg.discount)
        # 训练次数
        infer_until_step = utils.Until(self.cfg.agent.num_inference_steps,
                                       self.cfg.action_repeat)
        # 初始化
        temp_step = 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        replay_storage.add(time_step, meta)
        while infer_until_step(temp_step):
            if time_step.last():
                # 重启环境
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                replay_storage.add(time_step, meta)
            # 获取技能向量
            # meta = self.agent.update_meta(meta, temp_step, time_step)
            meta = self.agent.init_meta()
            # 采样
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, meta, temp_step, eval_mode=False, infer=True)
            # 环境交互
            time_step = self.train_env.step(action)
            replay_storage.add(time_step, meta)
            temp_step += 1
        replay_iter = iter(replay_loader)
        meta = self.agent.infer_meta(replay_iter, self.global_step)
        # meta = self.agent.infer_meta(self.replay_iter, self.global_step)
        self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
        self.evaluate_train()
        
    def train(self):
        self.evaluate_diversity()
        self. infer_meta()
        # 训练次数
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        test_every_step = utils.Every(10000,
                                      self.cfg.action_repeat)
        # 初始化
        episode_step, episode_reward, success = 0, 0, 0
        time_step = self.train_env.reset()
        if self.domain=='jaco':
            episode_step += 1
            self._global_step += 1  
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self.domain == 'metaworld' or self.domain == 'metaworld1':
                    success += time_step.info['success']      
                # 记录
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('success_ratio', success)
                        log('step', self.global_step)
                # 评估
                if test_every_step(self.global_step):
                    self.infer_meta()
                if eval_every_step(self.global_step):
                    self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
                    self.evaluate()
                    self.evaluate_train()
                # 重启环境
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                # 保存权重
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                if self.domain=='jaco':
                    episode_step += 1
                    self._global_step += 1  
                episode_step = 0
                episode_reward = 0   
                success=0
            # # 采样
            meta = self.agent.update_meta(meta, self.global_step, time_step)
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation, meta, self.global_step, eval_mode=False)
            # 更新agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            # 环境交互
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            episode_step += 1
            self._global_step += 1       

    def save_snapshot(self):
        snapshot = Path(self.model_dir) / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
@hydra.main(config_path='.', config_name='my_pretrain_new')
def main(cfg):
    from my_pretrain_new_fb import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    # if snapshot.exists():
    #     print(f'resuming: {snapshot}')
    #     workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
