import pdb
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from collections import OrderedDict
from dm_env import specs

import utils
import typing as tp
import goals as _goals
from agent.ddpg import DDPGAgent
from agent.fb_modules import IdentityMap
from agent.fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap2, OnlineCov
from agent.ddpg import Critic, Critic_C
from agent.ddpg import Actor as DDPG_Actor
from agent.rnd import RNDAgent

class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.skill_dim))

        self.next_state_net = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.skill_dim))

        self.pred_net = nn.Sequential(nn.Linear(2 * self.skill_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.skill_dim))

        self.skill_forward_net = nn.Sequential(nn.Linear(2 * self.skill_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.skill_dim))

        if project_skill:
            self.skill_net = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                            nn.Linear(hidden_dim, self.skill_dim))
        else:
            self.skill_net = nn.Identity()  
   
        self.apply(utils.weight_init)

    def forward(self,state,next_state,skill):
        assert len(state.size()) == len(next_state.size())
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state,next_state],1))
        return query, key

    def skill_forward(self, obs, skill, next_obs):
        with torch.no_grad():
            obs = self.state_net(obs)
            next_obs = self.state_net(next_obs)
        next_skill_obs_hat = self.skill_forward_net(torch.cat([obs.detach(), skill], dim=-1))
        skill_forward_error = torch.norm(next_obs.detach() - next_skill_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        return skill_forward_error
    
    def get_skill_forward(self, obs_rep, skill):
        next_obs_hat = self.skill_forward_net(torch.cat([obs_rep, skill], dim=-1))
        return next_obs_hat
    
class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S

class APTArgs:
    def __init__(self,knn_k=16,knn_avg=True, rms=True,knn_clip=0.0005,):
        self.knn_k = knn_k 
        self.knn_avg = knn_avg 
        self.rms = rms 
        self.knn_clip = knn_clip


def compute_apt_reward(source, target, rms, args):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)
    reward, _ = sim_matrix.topk(args.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(source.device))  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(source.device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward

class RND(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 clip_val=5.):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.predictor = nn.Sequential(encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(copy.deepcopy(encoder),
                                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error
    
# 原版fb_fixed6+rnd拆分
class dvfbAgent:
    def __init__(self, name, skill_dim, reward_free, obs_type, obs_shape, action_shape, device, lr, lr_coef, fb_target_tau, \
                 update_every_steps, use_tb, use_wandb, num_expl_steps, num_inference_steps, hidden_dim, backward_hidden_dim, \
                 feature_dim, z_dim, stddev_schedule, stddev_clip, update_z_every_step, update_z_proba, nstep, batch_size, init_fb, \
                 update_encoder, goal_space, ortho_coef, log_std_bounds, temp, boltzmann, debug, future_ratio, mix_ratio, rand_weight, \
                 preprocess, norm_z, q_loss, q_loss_coef, additional_metric, add_trunk):
        # 参数
        self.name = name
        self.skill_dim = skill_dim
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.lr = lr
        self.lr_coef = lr_coef
        self.fb_target_tau = fb_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.num_inference_steps = num_inference_steps
        self.hidden_dim = hidden_dim
        self.backward_hidden_dim = backward_hidden_dim
        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.update_z_every_step = update_z_every_step
        self.update_z_proba = update_z_proba
        self.nstep = nstep
        self.batch_size = batch_size
        self.init_fb = init_fb
        self.update_encoder = update_encoder
        self.goal_space = goal_space
        self.ortho_coef = ortho_coef
        self.log_std_bounds = log_std_bounds
        self.temp = temp
        self.boltzmann = boltzmann
        self.debug = debug
        self.future_ratio = future_ratio
        self.mix_ratio = mix_ratio
        self.rand_weight = rand_weight
        self.preprocess = preprocess
        self.norm_z = norm_z
        self.q_loss = q_loss
        self.q_loss_coef = q_loss_coef
        self.additional_metric = additional_metric
        self.add_trunk = add_trunk
        # 动作维度
        self.action_dim = action_shape[0]
        self.solved_meta = None
        # models
        if self.obs_type == 'pixels':
            self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
            self.encoder: nn.Module = Encoder(self.obs_shape).to(self.device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = self.obs_shape[0]
        if self.feature_dim < self.obs_dim:
            print(f"feature_dim {self.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
        goal_dim = self.obs_dim
        if self.goal_space is not None:
            goal_dim = _goals.get_goal_space_dim(self.goal_space)
        if self.z_dim < goal_dim:
            print(f"z_dim {self.z_dim} should not be smaller that goal_dim {goal_dim}")
        # create the network
        if self.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(self.obs_dim, self.z_dim, self.action_dim,
                                                      self.hidden_dim, self.log_std_bounds).to(self.device)
        else:
            self.actor = Actor(self.obs_dim, self.z_dim, self.action_dim, self.feature_dim, self.hidden_dim,
                               preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
        # critic
        self.critic = Critic(obs_type, self.obs_dim+self.z_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim+self.z_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # forward
        self.forward_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
                                      self.feature_dim, self.hidden_dim,
                                      preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
        # backward
        if self.debug:
            self.backward_net: nn.Module = IdentityMap().to(self.device)
            self.backward_target_net: nn.Module = IdentityMap().to(self.device)
        else:
            self.backward_net = BackwardMap2(goal_dim, self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
            self.backward_target_net = BackwardMap2(goal_dim,
                                                   self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
        # build up the target network
        self.forward_target_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
                                             self.feature_dim, self.hidden_dim,
                                             preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
        # load the weights into the target networks
        self.forward_target_net.load_state_dict(self.forward_net.state_dict())
        self.backward_target_net.load_state_dict(self.backward_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if self.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
        # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
        self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
                                        {'params': self.backward_net.parameters(), 'lr': self.lr_coef * self.lr}],
                                       lr=self.lr)

        self.train()
        self.critic_target.train()
        self.forward_target_net.train()
        self.backward_target_net.train()
        self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
        # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
        # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
        # self.online_cov.train()
        # self.trunk = nn.Sequential(
        #         nn.Linear(self.obs_dim + self.action_dim, hidden_dim),
        #         nn.LayerNorm(hidden_dim), nn.Tanh(),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(hidden_dim, 1)
        #         ).to(self.device)
        # self.trunk_opt = torch.optim.Adam(self.trunk.parameters(), lr=lr)
        # self.trunk.train()  
        ########################################################################
        # icm module
        self.sample_num = 50
        self.rep_dim = 512
        self.knn_clip=0.0
        self.knn_k=12
        self.knn_avg=True
        self.knn_rms=False
        project_skill = True
        self.rms = RMS(epsilon=1e-4, shape=(1,), device=self.device)
        self.cic = CIC(self.obs_dim, self.skill_dim, self.hidden_dim, project_skill).to(self.device)
        # optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(), lr=self.lr)
        self.cic.train()
        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE_NEW_DOT(rms, self.knn_clip, self.knn_k, self.knn_avg, self.knn_rms, self.device)
        # rnd module
        self.rnd_scale = 1.
        self.rnd_rep_dim=512
        self.rnd = RND(self.obs_dim, self.hidden_dim, self.rnd_rep_dim, self.encoder, self.aug, self.obs_shape, self.obs_type).to(self.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)
        self.reward_rms = utils.RMS(device=self.device)
        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.lr)
        self.rnd.train()  
        self.fb_rms = utils.RMS(device=self.device)
        self.extr_rms = utils.RMS(device=self.device)
        self.infer_rms = utils.RMS(device=self.device)
         
    # 加载模型
    def init_from(self, other) -> None:
        print("All attributes of 'other':", dir(other))
        # copy parameters over
        names = ["encoder", "actor"]
        if self.init_fb:
            names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
        for name in names:
            print("name", name)
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        # utils.hard_update_params(other.rnd_agent.actor, self.rnd_agent.actor)
        # for key, val in self.__dict__.items():
        #     if key not in ['critic_opt']:
        #         if isinstance(val, torch.optim.Optimizer):
        #             val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def train(self, training=True):
        self.training = training
        for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
            net.train(training)

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    # 采样技能向量z
    def sample_z(self, size, device="cpu"):
        gaussian_rdv = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
        if self.norm_z:
            z = math.sqrt(self.z_dim) * gaussian_rdv
        else:
            uniform_rdv = torch.rand((size, self.z_dim), dtype=torch.float32, device=device)
            z = np.sqrt(self.z_dim) * uniform_rdv * gaussian_rdv
        return z

    # 初始化技能向量
    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        else:
            z = self.sample_z(1)
            z = z.squeeze().numpy()
            meta = OrderedDict()
            meta['skill'] = z
        return meta

    # 更新技能向量
    def update_meta(self, meta, step, time_step):
        if step % self.update_z_every_step == 0 and np.random.rand() < self.update_z_proba:
            return self.init_meta()
        return meta

    # 根据goal点推理技能向量
    def get_goal_meta(self, goal_array: np.ndarray):
        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.backward_net(desired_goal)
        if self.norm_z:
            z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['z'] = z
        return meta

    # 推断技能向量
    def infer_meta(self, replay_iter, step):
        obs_list, next_obs_list, reward_list = [], [], []
        batch_size = 0
        while batch_size < self.num_inference_steps:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
            next_goal = future_obs
            obs_list.append(obs)
            next_obs_list.append(next_obs if self.goal_space is not None else next_obs)
            ####################################################
            # # # jaco
            # infer_mean, infer_var = self.infer_rms(reward)
            # # reward /= infer_var
            # reward = (reward - infer_mean) / infer_var
            ####################################################
            reward_list.append(reward)
            batch_size += next_obs.size(0)
        obs, next_obs, reward = torch.cat(obs_list, 0), torch.cat(next_obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, next_obs, reward = obs[:self.num_inference_steps], next_obs[:self.num_inference_steps], reward[:self.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, next_obs, reward)

    # 具体推断过程
    def infer_meta_from_obs_and_rewards(self, obs, next_obs, reward):
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        # # filter out small reward
        idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
        obs = obs[idx]
        next_obs = next_obs[idx]
        reward = reward[idx]
        with torch.no_grad():
            B = self.backward_net(obs, next_obs)
        # reward = torch.where(reward>torch.quantile(reward, 0.9), (reward-torch.quantile(reward, 0.9))/(reward.max()-torch.quantile(reward, 0.9)), 0)
        z = torch.matmul(reward.T, B) / reward.shape[0]
        if self.norm_z:
            z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
        meta = OrderedDict()
        meta['skill'] = z.squeeze().cpu().numpy()
        self.solved_meta = meta
        return meta

    # 执行动作
    def act(self, obs, meta, step, eval_mode, infer=False):
        # # 如果推断重新生成技能向量
        if infer:                                 
            meta = self.init_meta()
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['skill'], device=self.device).unsqueeze(0)  # type: ignore
        # 如果预训练采样使用探索智能体 
        if self.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            # if step < self.num_expl_steps and not infer:
            #     action.uniform_(-1.0, 1.0)
            if self.reward_free or infer:
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    # 更新fb，包括保证FB满足度量损失、Fz满足Q损失和正则化损失三项
    def update_fb(self, obs, action, discount, reward, next_obs, next_goal, z, step):
        metrics = {}
        # compute target successor measure
        with torch.no_grad():
            if self.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
            target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
            target_B = self.backward_target_net(next_obs, next_goal)  # batch x z_dim
            target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
            target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
            target_M = torch.min(target_M1, target_M2)

        # compute FB loss
        F1, F2 = self.forward_net(obs, z, action)
        B = self.backward_net(obs, next_goal)
        M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
        M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
        I = torch.eye(*M1.size(), device=M1.device)
        off_diag = ~I.bool()
        fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
        fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
        fb_loss = fb_offdiag + fb_diag

        # Q LOSS
        if self.q_loss:
            with torch.no_grad():
                next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
                next_Q = torch.min(next_Q1, nextQ2)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
                target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            fb_loss += self.q_loss_coef * q_loss

        # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.ortho_coef * orth_loss

        # Cov = torch.cov(B.T)  # Vicreg loss
        # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
        # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
        # orth_loss =  var_loss + cov_loss
        # fb_loss += self.cfg.ortho_coef * orth_loss

        if self.use_tb or self.use_wandb or self.use_hiplog:
            metrics['target_M'] = target_M.mean().item()
            metrics['M1'] = M1.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['B'] = B.mean().item()
            metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['fb_loss'] = fb_loss.item()
            metrics['fb_diag'] = fb_diag.item()
            metrics['fb_offdiag'] = fb_offdiag.item()
            if self.q_loss:
                metrics['critic_target_q'] = target_Q.mean().item()
                metrics['critic_q1'] = Q1.mean().item()
                metrics['critic_q2'] = Q2.mean().item()
                metrics['critic_loss'] = q_loss.item()
            metrics['orth_loss'] = orth_loss.item()
            metrics['orth_loss_diag'] = orth_loss_diag.item()
            metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
            eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
            metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
            metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
            if isinstance(self.fb_opt, torch.optim.Adam):
                metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

        # optimize FB
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.fb_opt.zero_grad(set_to_none=True)
        fb_loss.backward()
        self.fb_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_critic(self, obs, action, z, reward, discount, next_obs, step):
        metrics = dict()

        # 70-72，80-82没有
        # finetuine时奖励归一化
        with torch.no_grad():
            if not self.reward_free:
                B = self.backward_net(obs, next_obs)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1).view(obs.shape[0], -1)
                fb_reward_mean, fb_reward_var = self.fb_rms(implicit_reward)
                extr_reward_mean, extr_reward_var = self.extr_rms(reward)
                reward = reward * fb_reward_mean / (extr_reward_mean+1e-8)
                # reward = fb_reward_mean + (reward - extr_reward_mean) * fb_reward_var / extr_reward_var
                # print("fb_reward_mean", fb_reward_mean)
                # print("extr_reward_mean", extr_reward_mean)
                # print("reward", reward.mean().item())
                # if self.reward_free:
                #     if self.use_tb or self.use_wandb:
                #         metrics['z_extr_reward_mean'] = extr_reward_mean.item()
                #         metrics['z_fb_reward_mean'] = fb_reward_mean.item()
                #         metrics['z_reward'] = reward.mean().item()
        
        with torch.no_grad():
            if self.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
            next_critic_obs = torch.cat([next_obs, z], dim=-1)
            target_Q1, target_Q2 = self.critic_target(next_critic_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        critic_obs = torch.cat([obs, z], dim=-1)
        Q1, Q2 = self.critic(critic_obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.reward_free:
            if self.use_tb or self.use_wandb:
                metrics['intinsic_critic_target_q'] = target_Q.mean().item()
                metrics['intinsic_critic_q1'] = Q1.mean().item()
                metrics['intinsic_critic_q2'] = Q2.mean().item()
                metrics['intinsic_critic_loss'] = critic_loss.item()
        else:
            if self.use_tb or self.use_wandb:
                metrics['critic_target_q'] = target_Q.mean().item()
                metrics['critic_q1'] = Q1.mean().item()
                metrics['critic_q2'] = Q2.mean().item()
                metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, z, step):
        metrics: tp.Dict[str, float] = {}
        if self.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        if self.reward_free:
            F1, F2 = self.forward_net(obs, z, action)
            MQ1 = torch.einsum('sd, sd -> s', F1, z).view(obs.shape[0], -1)
            MQ2 = torch.einsum('sd, sd -> s', F2, z).view(obs.shape[0], -1)
            critic_obs = torch.cat([obs, z], dim=-1)
            Q1, Q2 = self.critic(critic_obs, action)
            with torch.no_grad():
                alpha = torch.min(MQ1, MQ2).mean().item()/torch.min(Q1, Q2).mean().item()
            Q = 2*alpha*torch.min(Q1, Q2)+torch.min(MQ1, MQ2)
        else:
            F1, F2 = self.forward_net(obs, z, action)
            MQ1 = torch.einsum('sd, sd -> s', F1, z).view(obs.shape[0], -1)
            MQ2 = torch.einsum('sd, sd -> s', F2, z).view(obs.shape[0], -1)
            critic_obs = torch.cat([obs, z], dim=-1)
            Q1, Q2 = self.critic(critic_obs, action)            
            Q = torch.min(Q1, Q2)
            MQ = torch.min(MQ1, MQ2)
            # if step > 50000:
            #     alpha=0
            # else:
            #     alpha = 1-step/50000
            # Q = Q + 0.5*alpha*MQ 
            #########################
            # # 正规的
            # Q = Q + 0.5*MQ 
            
            
            # # 201-203
            # Q = Q + 0.2*MQ 
            # # 211-213
            # Q = Q + 0.05*MQ 
            # # 221-223
            # Q = Q + MQ 
            # 231-233
            alpha = (Q / MQ).detach()
            Q = Q + alpha * MQ 
            
            ########
            # 70-72 纯Q
            # 啥都没有
            
            # alpha = 1-step/110000
            # Q = Q + 0.2*alpha*MQ 
            # Q = 0.05*(Q + 0.2*MQ)
            # Q = 0.1*(Q + 0.2*MQ)
            
        if self.use_tb or self.use_wandb:
            metrics['actor_MQ'] = torch.min(MQ1, MQ2).mean().item()
            metrics['actor_Q'] = torch.min(Q1, Q2).mean().item()
        actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            if self.reward_free:
                metrics['actor_loss'] = actor_loss.item()
                metrics['q'] = Q.mean().item()
                if self.additional_metric:
                    metrics['q1_success'] = q1_success.float().mean().item()
                metrics['actor_logprob'] = log_prob.mean().item()
            else:
                metrics['actor_loss'] = actor_loss.item()
                metrics['actor_logprob'] = log_prob.mean().item()
                metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def compute_cpc_loss(self,obs,next_obs,skill):
        temperature = self.temp
        eps = 1e-6
        query, key = self.cic.forward(obs,next_obs,skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query,key.T) # (b,b)
        sim = torch.exp(cov / temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        return loss, cov / temperature

    def update_cic(self, obs, skill, next_obs, step):
        metrics = dict()
        
        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        skill_forward_error = self.cic.skill_forward(obs, skill, next_obs)
        loss = loss.mean() + skill_forward_error.mean()
        self.cic_optimizer.zero_grad()
        loss.backward()
        self.cic_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics['cic_loss'] = loss.item()
            metrics['cic_logits'] = logits.norm()

        return metrics
    
    def update_rnd(self, obs, step):
        metrics = dict()

        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        self.rnd_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['rnd_loss'] = loss.item()

        return metrics
    
    def compute_intr_reward(self, obs, skill, next_obs, metrics, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward1 = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        args = APTArgs()
        B, D = obs.size()
        obs = self.cic.state_net(obs)
        next_obs = self.cic.state_net(next_obs)
        key = self.cic.pred_net(torch.cat([obs,next_obs],1))
        query = self.cic.skill_net(skill)
        query = query.unsqueeze(0).repeat(B, 1, 1)
        pbe_reward = self.pbe(key, query)
        metrics['intr_pbe_reward'] = pbe_reward.mean().item()
        metrics['intr_reward1'] = reward1.mean().item()
        reward = 0.5*pbe_reward + reward1
        # reward = 0.1*pbe_reward + reward1
        return reward # (b,1)

    def compute_rnd_intr_reward(self, obs, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward
    
    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics
            
        # 获取经验池
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
        next_goal = future_obs
        if self.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal


        if self.reward_free:
            
            # 采样技能向量
            z = self.sample_z(self.batch_size, device=self.device)
            if not z.shape[-1] == self.z_dim:
                raise RuntimeError("There's something wrong with the logic here")

            # 后向模型输入
            backward_input = obs
            future_goal = future_obs
            if self.goal_space is not None:
                assert batch.goal is not None
                backward_input = goal
                future_goal = future_goal
            perm = torch.randperm(self.batch_size)
            backward_input = backward_input[perm]

            # 产生z
            if self.mix_ratio > 0:
                mix_idxs = np.where(np.random.uniform(size=self.batch_size) < self.mix_ratio)[0]
                if not self.rand_weight:
                    with torch.no_grad():
                        mix_z = self.backward_net(obs[mix_idxs], backward_input[mix_idxs]).detach()
                else:
                    # generate random weight
                    weight = torch.rand(size=(mix_idxs.shape[0], self.batch_size)).to(self.device)
                    weight = F.normalize(weight, dim=1)
                    uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.device)
                    weight = uniform_rdv * weight
                    with torch.no_grad():
                        mix_z = torch.matmul(weight, self.backward_net(obs, backward_input).detach())
                if self.norm_z:
                    mix_z = math.sqrt(self.z_dim) * F.normalize(mix_z, dim=1)
                z[mix_idxs] = mix_z

            # hindsight replay
            if self.future_ratio > 0:
                assert future_goal is not None
                future_idxs = np.where(np.random.uniform(size=self.batch_size) < self.future_ratio)
                z[future_idxs] = self.backward_net(obs[future_idxs], future_goal[future_idxs]).detach()

            ###########################################################################
            # 更新rnd智能体
            metrics.update(self.update_rnd(obs, step))
            # 更新icm智能体
            intrinsic_skill = self.backward_net(obs, next_obs).detach()
            metrics.update(self.update_cic(obs, intrinsic_skill, next_obs, step))
            # 计算内在奖励
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(obs, intrinsic_skill, next_obs, metrics, step)
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward

            metrics.update(self.update_critic(obs=obs, action=action, discount=discount,
                                        next_obs=next_obs, reward=reward, z=z, step=step))
            
            metrics.update(self.update_fb(obs=obs, action=action, discount=discount, reward=reward,
                                        next_obs=next_obs, next_goal=next_goal, z=z, step=step))
        
        else:
            z=skill
            ####################################################
            # # 更新rnd智能体
            # metrics.update(self.update_rnd(obs, step))
            # # 计算内在奖励
            # with torch.no_grad():
            #     intr_reward = self.compute_rnd_intr_reward(obs, step)
            #     metrics['intr_reward'] = intr_reward.mean().item()
            # reward += 0.1*intr_reward
            #####################################################
            metrics.update(self.update_critic(obs=obs, action=action, discount=discount,
                                        next_obs=next_obs, reward=reward, z=z, step=step))
        
        # update actor
        metrics.update(self.update_actor(obs, z, step))

        # update critic target
        utils.soft_update_params(self.forward_net, self.forward_target_net,
                                 self.fb_target_tau)
        utils.soft_update_params(self.backward_net, self.backward_target_net,
                                 self.fb_target_tau)
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.fb_target_tau)
        
        return metrics



    # def update_critic(self, obs, action, z, reward, discount, next_obs, step):
    #     metrics = dict()

    #     with torch.no_grad():
    #         if self.boltzmann:
    #             dist = self.actor(next_obs, z)
    #             next_action = dist.sample()
    #         else:
    #             stddev = utils.schedule(self.stddev_schedule, step)
    #             dist = self.actor(next_obs, z, stddev)
    #             next_action = dist.sample(clip=self.stddev_clip)
    #         next_critic_obs = torch.cat([next_obs, z], dim=-1)
    #         target_Q1, target_Q2 = self.critic_target(next_critic_obs, next_action)
    #         target_V = torch.min(target_Q1, target_Q2)
            
    #         B = self.backward_net(obs, next_obs)
    #         cov = torch.matmul(B.T, B) / B.shape[0]
    #         inv_cov = torch.inverse(cov)
    #         implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1).view(obs.shape[0], -1)
    #         # print("implicit_reward", implicit_reward.shape)
    #         fb_reward_mean, fb_reward_var = self.fb_rms(implicit_reward)
    #         extr_reward_mean, extr_reward_var = self.extr_rms(reward)
            
    #         # reward = reward * implicit_reward.mean().item() / reward.mean().item()
    #         # print("fb_reward_mean", fb_reward_mean.shape)
    #         # print("extr_reward_mean", extr_reward_mean.shape)
    #         reward = reward * fb_reward_mean / extr_reward_mean
    #         # print("torch.sqrt(fb_reward_var)", torch.sqrt(fb_reward_var))
    #         # print("torch.sqrt(extr_reward_var)", torch.sqrt(extr_reward_var))
    #         # reward = fb_reward_mean + torch.sqrt(fb_reward_var) * (reward - extr_reward_mean) / torch.sqrt(extr_reward_var)
    #         target_Q = reward + (discount * target_V)
        
    #     critic_obs = torch.cat([obs, z], dim=-1)
    #     Q1, Q2 = self.critic(critic_obs, action)
    #     critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

    #     if self.reward_free:
    #         if self.use_tb or self.use_wandb:
    #             metrics['intinsic_critic_target_q'] = target_Q.mean().item()
    #             metrics['intinsic_critic_q1'] = Q1.mean().item()
    #             metrics['intinsic_critic_q2'] = Q2.mean().item()
    #             metrics['intinsic_critic_loss'] = critic_loss.item()
    #     else:
    #         if self.use_tb or self.use_wandb:
    #             metrics['critic_target_q'] = target_Q.mean().item()
    #             metrics['critic_q1'] = Q1.mean().item()
    #             metrics['critic_q2'] = Q2.mean().item()
    #             metrics['critic_loss'] = critic_loss.item()

    #     # optimize critic
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.zero_grad(set_to_none=True)
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()
    #     self.critic_opt.step()
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.step()
    #     return metrics   
    
    
    # def update_actor(self, obs, z, step):
    #     metrics: tp.Dict[str, float] = {}
    #     if self.boltzmann:
    #         dist = self.actor(obs, z)
    #         action = dist.rsample()
    #     else:
    #         stddev = utils.schedule(self.stddev_schedule, step)
    #         dist = self.actor(obs, z, stddev)
    #         action = dist.sample(clip=self.stddev_clip)

    #     log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
    #     if self.reward_free:
    #         F1, F2 = self.forward_net(obs, z, action)
    #         MQ1 = torch.einsum('sd, sd -> s', F1, z).view(obs.shape[0], -1)
    #         MQ2 = torch.einsum('sd, sd -> s', F2, z).view(obs.shape[0], -1)
    #         critic_obs = torch.cat([obs, z], dim=-1)
    #         Q1, Q2 = self.critic(critic_obs, action)
    #         # print("torch.min(Q1, Q2)", torch.min(Q1, Q2).mean())
    #         # print("torch.min(MQ1, MQ2)", torch.min(MQ1, MQ2).mean())
    #         with torch.no_grad():
    #             alpha = torch.min(MQ1, MQ2).mean().item()/torch.min(Q1, Q2).mean().item()
    #         Q = 2*alpha*torch.min(Q1, Q2)+torch.min(MQ1, MQ2)
    #     else:
    #         F1, F2 = self.forward_net(obs, z, action)
    #         MQ1 = torch.einsum('sd, sd -> s', F1, z).view(obs.shape[0], -1)
    #         MQ2 = torch.einsum('sd, sd -> s', F2, z).view(obs.shape[0], -1)
    #         critic_obs = torch.cat([obs, z], dim=-1)
    #         Q1, Q2 = self.critic(critic_obs, action)
    #         # print("torch.min(Q1, Q2)", torch.min(Q1, Q2).mean())
    #         # print("torch.min(MQ1, MQ2)", torch.min(MQ1, MQ2).mean())
    #         # Q = torch.min(Q1, Q2)+0.01*torch.min(MQ1, MQ2)
            
    #         Q = torch.min(Q1, Q2)
    #         MQ = torch.min(MQ1, MQ2)
    #         # if step>50000:
    #         #     alpha=1
    #         # else:
    #         #     alpha=0
    #         # alpha = 10*step/100000
    #         # Q = MQ #0.1*(MQ+alpha*Q)  
    #         # Q = 0.1*torch.where(Q>50, MQ+2*Q, MQ+5*Q)  
    #         # Q = 0.1*torch.where(Q>100, MQ+2*Q, MQ+5*Q)  
    #         # Q = 0.1*torch.where(Q>50, MQ+5*Q, MQ+10*Q)             
            
    #         # seed 81:0.1* (无0.1*critic loss) 82\83:1*
    #         # with torch.no_grad():
    #         #     alpha = MQ/Q 
    #         # Q = torch.where(MQ > 5*Q, MQ+5*Q, MQ+ alpha*Q)     
    #         Q = 0.1*torch.where(Q>50, MQ+2*Q, MQ+5*Q)         
            
    #     if self.use_tb or self.use_wandb:
    #         metrics['actor_MQ'] = torch.min(MQ1, MQ2).mean().item()
    #         metrics['actor_Q'] = torch.min(Q1, Q2).mean().item()
    #     actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

    #     # optimize actor
    #     self.actor_opt.zero_grad(set_to_none=True)
    #     actor_loss.backward()
    #     self.actor_opt.step()

    #     if self.use_tb or self.use_wandb:
    #         if self.reward_free:
    #             metrics['actor_loss'] = actor_loss.item()
    #             metrics['q'] = Q.mean().item()
    #             if self.additional_metric:
    #                 metrics['q1_success'] = q1_success.float().mean().item()
    #             metrics['actor_logprob'] = log_prob.mean().item()
    #         else:
    #             metrics['actor_loss'] = actor_loss.item()
    #             metrics['actor_logprob'] = log_prob.mean().item()
    #             metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
    #         # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

    #     return metrics



############################################
# 1. 直接使用fb的价值函数finetune
    # def update_critic(self, obs, action, z, reward, discount, next_obs, step):
    #     metrics = dict()

    #     with torch.no_grad():
    #         if self.boltzmann:
    #             dist = self.actor(next_obs, z)
    #             next_action = dist.sample()
    #         else:
    #             stddev = utils.schedule(self.stddev_schedule, step)
    #             dist = self.actor(next_obs, z, stddev)
    #             next_action = dist.sample(clip=self.stddev_clip)
    #         target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
    #         target_Q1, target_Q2 = [torch.einsum('sd, sd -> s', target_Fi, z).view(obs.shape[0], -1) for target_Fi in [target_F1, target_F2]]
    #         target_V = torch.min(target_Q1, target_Q2)
    #         target_Q = reward + (discount * target_V)
            
    #     F1, F2 = self.forward_net(obs, z, action)
    #     Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z).view(obs.shape[0], -1) for Fi in [F1, F2]]
    #     critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

    #     if self.use_tb or self.use_wandb:
    #         metrics['critic_target_q'] = target_Q.mean().item()
    #         metrics['critic_q1'] = Q1.mean().item()
    #         metrics['critic_q2'] = Q2.mean().item()
    #         metrics['critic_loss'] = critic_loss.item()

    #     # optimize critic
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.zero_grad(set_to_none=True)
    #     self.fb_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()
    #     self.fb_opt.step()
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.step()
    #     return metrics
    
    # def update_actor(self, obs, z, step):
    #     metrics: tp.Dict[str, float] = {}
    #     if self.boltzmann:
    #         dist = self.actor(obs, z)
    #         action = dist.rsample()
    #     else:
    #         stddev = utils.schedule(self.stddev_schedule, step)
    #         dist = self.actor(obs, z, stddev)
    #         action = dist.sample(clip=self.stddev_clip)

    #     log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
    #     F1, F2 = self.forward_net(obs, z, action)
    #     Q1 = torch.einsum('sd, sd -> s', F1, z).view(obs.shape[0], -1)
    #     Q2 = torch.einsum('sd, sd -> s', F2, z).view(obs.shape[0], -1)
    #     if self.additional_metric:
    #         q1_success = Q1 > Q2
    #     Q = torch.min(Q1, Q2)

    #     actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

    #     # optimize actor
    #     self.actor_opt.zero_grad(set_to_none=True)
    #     actor_loss.backward()
    #     self.actor_opt.step()

    #     if self.use_tb or self.use_wandb:
    #         if self.reward_free:
    #             metrics['actor_loss'] = actor_loss.item()
    #             metrics['q'] = Q.mean().item()
    #             if self.additional_metric:
    #                 metrics['q1_success'] = q1_success.float().mean().item()
    #             metrics['actor_logprob'] = log_prob.mean().item()
    #         else:
    #             metrics['actor_loss'] = actor_loss.item()
    #             metrics['actor_logprob'] = log_prob.mean().item()
    #             metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
    #         # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

    #     return metrics


#################################################################
# FB+Q
    # def update_critic(self, obs, action, z, reward, discount, next_obs, step):
    #     metrics = dict()

    #     with torch.no_grad():
    #         if self.boltzmann:
    #             dist = self.actor(next_obs, z)
    #             next_action = dist.sample()
    #         else:
    #             stddev = utils.schedule(self.stddev_schedule, step)
    #             dist = self.actor(next_obs, z, stddev)
    #             next_action = dist.sample(clip=self.stddev_clip)
    #         target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    #         target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
    #         target_MQ1, target_MQ2 = [torch.einsum('sd, sd -> s', target_Fi, z).view(obs.shape[0], -1) for target_Fi in [target_F1, target_F2]]
    #         target_V = torch.min(target_Q1+target_MQ1, target_Q1+target_MQ2)
    #         target_Q = reward + (discount * target_V)
            
    #     F1, F2 = self.forward_net(obs, z, action)
    #     MQ1, MQ2 = [torch.einsum('sd, sd -> s', Fi, z).view(obs.shape[0], -1) for Fi in [F1, F2]]     
    #     Q1, Q2 = self.critic(obs, action)
    #     Q1, Q2 = Q1+MQ1, Q1+MQ2
    #     critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

    #     if self.use_tb or self.use_wandb:
    #         metrics['critic_target_q'] = target_Q.mean().item()
    #         metrics['critic_q1'] = Q1.mean().item()
    #         metrics['critic_q2'] = Q2.mean().item()
    #         metrics['critic_loss'] = critic_loss.item()

    #     # optimize critic
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.zero_grad(set_to_none=True)
    #     self.critic_opt.zero_grad(set_to_none=True)
    #     critic_loss.backward()
    #     self.critic_opt.step()
    #     if self.encoder_opt is not None:
    #         self.encoder_opt.step()
    #     return metrics
    
    # def update_actor(self, obs, z, step):
    #     metrics: tp.Dict[str, float] = {}
    #     if self.boltzmann:
    #         dist = self.actor(obs, z)
    #         action = dist.rsample()
    #     else:
    #         stddev = utils.schedule(self.stddev_schedule, step)
    #         dist = self.actor(obs, z, stddev)
    #         action = dist.sample(clip=self.stddev_clip)

    #     log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
    #     if self.reward_free:
    #         F1, F2 = self.forward_net(obs, z, action)
    #         Q1 = torch.einsum('sd, sd -> s', F1, z)
    #         Q2 = torch.einsum('sd, sd -> s', F2, z)
    #         if self.additional_metric:
    #             q1_success = Q1 > Q2
    #         Q = torch.min(Q1, Q2)
    #     else:
    #         F1, F2 = self.forward_net(obs, z, action)
    #         MQ1 = torch.einsum('sd, sd -> s', F1, z).view(obs.shape[0], -1)
    #         MQ2 = torch.einsum('sd, sd -> s', F2, z).view(obs.shape[0], -1)
    #         Q1, Q2 = self.critic(obs, action)
    #         Q1, Q2 = Q1+MQ1, Q1+MQ2
    #         # print("torch.min(Q1, Q2)", torch.min(Q1, Q2).mean())
    #         # print("torch.min(MQ1, MQ2)", torch.min(MQ1, MQ2).mean())
    #         Q = torch.min(Q1, Q2)
    #         if self.use_tb or self.use_wandb:
    #             metrics['actor_MQ'] = torch.min(MQ1, MQ2).mean().item()
    #             metrics['actor_Q'] = torch.min(Q1, Q2).mean().item()
    #     actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

    #     # optimize actor
    #     self.actor_opt.zero_grad(set_to_none=True)
    #     actor_loss.backward()
    #     self.actor_opt.step()

    #     if self.use_tb or self.use_wandb:
    #         if self.reward_free:
    #             metrics['actor_loss'] = actor_loss.item()
    #             metrics['q'] = Q.mean().item()
    #             if self.additional_metric:
    #                 metrics['q1_success'] = q1_success.float().mean().item()
    #             metrics['actor_logprob'] = log_prob.mean().item()
    #         else:
    #             metrics['actor_loss'] = actor_loss.item()
    #             metrics['actor_logprob'] = log_prob.mean().item()
    #             metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
    #         # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

    #     return metrics


# # 修改Q finetune的更新方式
# class fb6Agent:
#     def __init__(self, name, skill_dim, reward_free, obs_type, obs_shape, action_shape, device, lr, lr_coef, fb_target_tau, \
#                  update_every_steps, use_tb, use_wandb, num_expl_steps, num_inference_steps, hidden_dim, backward_hidden_dim, \
#                  feature_dim, z_dim, stddev_schedule, stddev_clip, update_z_every_step, update_z_proba, nstep, batch_size, init_fb, \
#                  update_encoder, goal_space, ortho_coef, log_std_bounds, temp, boltzmann, debug, future_ratio, mix_ratio, rand_weight, \
#                  preprocess, norm_z, q_loss, q_loss_coef, additional_metric, add_trunk):
#         # 参数
#         self.name = name
#         self.skill_dim = skill_dim
#         self.reward_free = reward_free
#         self.obs_type = obs_type
#         self.obs_shape = obs_shape
#         self.action_shape = action_shape
#         self.device = device
#         self.lr = lr
#         self.lr_coef = lr_coef
#         self.fb_target_tau = fb_target_tau
#         self.update_every_steps = update_every_steps
#         self.use_tb = use_tb
#         self.use_wandb = use_wandb
#         self.num_expl_steps = num_expl_steps
#         self.num_inference_steps = num_inference_steps
#         self.hidden_dim = hidden_dim
#         self.backward_hidden_dim = backward_hidden_dim
#         self.feature_dim = feature_dim
#         self.z_dim = z_dim
#         self.stddev_schedule = stddev_schedule
#         self.stddev_clip = stddev_clip
#         self.update_z_every_step = update_z_every_step
#         self.update_z_proba = update_z_proba
#         self.nstep = nstep
#         self.batch_size = batch_size
#         self.init_fb = init_fb
#         self.update_encoder = update_encoder
#         self.goal_space = goal_space
#         self.ortho_coef = ortho_coef
#         self.log_std_bounds = log_std_bounds
#         self.temp = temp
#         self.boltzmann = boltzmann
#         self.debug = debug
#         self.future_ratio = future_ratio
#         self.mix_ratio = mix_ratio
#         self.rand_weight = rand_weight
#         self.preprocess = preprocess
#         self.norm_z = norm_z
#         self.q_loss = q_loss
#         self.q_loss_coef = q_loss_coef
#         self.additional_metric = additional_metric
#         self.add_trunk = add_trunk
#         # 动作维度
#         self.action_dim = action_shape[0]
#         self.solved_meta = None
#         # models
#         if self.obs_type == 'pixels':
#             self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
#             self.encoder: nn.Module = Encoder(self.obs_shape).to(self.device)
#             self.obs_dim = self.encoder.repr_dim
#         else:
#             self.aug = nn.Identity()
#             self.encoder = nn.Identity()
#             self.obs_dim = self.obs_shape[0]
#         if self.feature_dim < self.obs_dim:
#             print(f"feature_dim {self.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
#         goal_dim = self.obs_dim
#         if self.goal_space is not None:
#             goal_dim = _goals.get_goal_space_dim(self.goal_space)
#         if self.z_dim < goal_dim:
#             print(f"z_dim {self.z_dim} should not be smaller that goal_dim {goal_dim}")
#         # create the network
#         if self.boltzmann:
#             self.actor: nn.Module = DiagGaussianActor(self.obs_dim, self.z_dim, self.action_dim,
#                                                       self.hidden_dim, self.log_std_bounds).to(self.device)
#         else:
#             self.actor = Actor(self.obs_dim, self.z_dim, self.action_dim, self.feature_dim, self.hidden_dim,
#                                preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # critic
#         self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
#                              feature_dim, hidden_dim).to(device)
#         self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
#                                     feature_dim, hidden_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         # forward
#         self.forward_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
#                                       self.feature_dim, self.hidden_dim,
#                                       preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # backward
#         if self.debug:
#             self.backward_net: nn.Module = IdentityMap().to(self.device)
#             self.backward_target_net: nn.Module = IdentityMap().to(self.device)
#         else:
#             self.backward_net = BackwardMap(goal_dim, self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
#             self.backward_target_net = BackwardMap(goal_dim,
#                                                    self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
#         # build up the target network
#         self.forward_target_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
#                                              self.feature_dim, self.hidden_dim,
#                                              preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # load the weights into the target networks
#         self.forward_target_net.load_state_dict(self.forward_net.state_dict())
#         self.backward_target_net.load_state_dict(self.backward_net.state_dict())
#         # optimizers
#         self.encoder_opt: tp.Optional[torch.optim.Adam] = None
#         if self.obs_type == 'pixels':
#             self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
#         self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
#         # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
#         # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
#         self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
#                                         {'params': self.backward_net.parameters(), 'lr': self.lr_coef * self.lr}],
#                                        lr=self.lr)

#         self.train()
#         self.critic_target.train()
#         self.forward_target_net.train()
#         self.backward_target_net.train()
#         self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
#         # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
#         # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
#         # self.online_cov.train()

#         self.rnd_agent = RNDAgent(rnd_rep_dim=512, update_encoder=update_encoder, rnd_scale=1., name='rnd', reward_free=reward_free, obs_type=obs_type, \
#                                   obs_shape=obs_shape, action_shape=action_shape, device=device, lr=lr, feature_dim=50, hidden_dim=hidden_dim, \
#                                   critic_target_tau=fb_target_tau, num_expl_steps=num_expl_steps, update_every_steps=update_every_steps, \
#                                   stddev_schedule=stddev_schedule, nstep=3, batch_size=batch_size, stddev_clip=stddev_clip, init_critic=init_fb, \
#                                   use_tb=use_tb, use_wandb=use_wandb)
        
        
#     # 加载模型
#     def init_from(self, other) -> None:
#         print("All attributes of 'other':", dir(other))
#         # copy parameters over
#         names = ["encoder", "actor"]
#         if self.init_fb:
#             names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
#         for name in names:
#             print("name", name)
#             utils.hard_update_params(getattr(other, name), getattr(self, name))
#         utils.hard_update_params(other.rnd_agent.actor, self.rnd_agent.actor)
#         # for key, val in self.__dict__.items():
#         #     if key not in ['critic_opt']:
#         #         if isinstance(val, torch.optim.Optimizer):
#         #             val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

#     def train(self, training=True):
#         self.training = training
#         for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
#             net.train(training)

#     def get_meta_specs(self):
#         return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

#     # 采样技能向量z
#     def sample_z(self, size, device="cpu"):
#         gaussian_rdv = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
#         gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * gaussian_rdv
#         else:
#             uniform_rdv = torch.rand((size, self.z_dim), dtype=torch.float32, device=device)
#             z = np.sqrt(self.z_dim) * uniform_rdv * gaussian_rdv
#         return z

#     # 初始化技能向量
#     def init_meta(self):
#         if self.solved_meta is not None:
#             return self.solved_meta
#         else:
#             z = self.sample_z(1)
#             z = z.squeeze().numpy()
#             meta = OrderedDict()
#             meta['skill'] = z
#         return meta

#     # 更新技能向量
#     def update_meta(self, meta, step, time_step):
#         if step % self.update_z_every_step == 0 and np.random.rand() < self.update_z_proba:
#             return self.init_meta()
#         return meta

#     # 根据goal点推理技能向量
#     def get_goal_meta(self, goal_array: np.ndarray):
#         desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             z = self.backward_net(desired_goal)
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
#         z = z.squeeze(0).cpu().numpy()
#         meta = OrderedDict()
#         meta['z'] = z
#         return meta

#     # 推断技能向量
#     def infer_meta(self, replay_iter, step):
#         obs_list, reward_list = [], []
#         batch_size = 0
#         while batch_size < self.num_inference_steps:
#             batch = next(replay_iter)
#             obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
#             next_goal = next_obs
#             obs_list.append(next_goal if self.goal_space is not None else next_obs)
#             reward_list.append(reward)
#             batch_size += next_obs.size(0)
#         obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
#         obs, reward = obs[:self.num_inference_steps], reward[:self.num_inference_steps]
#         return self.infer_meta_from_obs_and_rewards(obs, reward)

#     # 具体推断过程
#     def infer_meta_from_obs_and_rewards(self, obs, reward):
#         print('max reward: ', reward.max().cpu().item())
#         print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
#         print('median reward: ', reward.median().cpu().item())
#         print('min reward: ', reward.min().cpu().item())
#         print('mean reward: ', reward.mean().cpu().item())
#         print('num reward: ', reward.shape[0])

#         # filter out small reward
#         # pdb.set_trace()
#         # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
#         # obs = obs[idx]
#         # reward = reward[idx]
#         with torch.no_grad():
#             B = self.backward_net(obs)
#         z = torch.matmul(reward.T, B) / reward.shape[0]
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
#         meta = OrderedDict()
#         meta['skill'] = z.squeeze().cpu().numpy()
#         self.solved_meta = meta
#         return meta

#     # 执行动作
#     def act(self, obs, meta, step, eval_mode, infer=False):
#         if infer:
#             meta = OrderedDict()
#             action = self.rnd_agent.act(obs, meta, step+4000, eval_mode)
#             return action
#         if (not eval_mode and self.reward_free):
#             meta = OrderedDict()
#             action = self.rnd_agent.act(obs, meta, step, eval_mode)
#             return action
#         else:
#             obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
#             h = self.encoder(obs)
#             z = torch.as_tensor(meta['skill'], device=self.device).unsqueeze(0)  # type: ignore
#             if self.boltzmann:
#                 dist = self.actor(h, z)
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(h, z, stddev)
#             if eval_mode:
#                 action = dist.mean
#                 if self.additional_metric:
#                     # the following is doing extra computation only used for metrics,
#                     # it should be deactivated eventually
#                     F_mean_s = self.forward_net(obs, z, action)
#                     # F_samp_s = self.forward_net(obs, z, dist.sample())
#                     F_rand_s = self.forward_net(obs, z, torch.zeros_like(action).uniform_(-1.0, 1.0))
#                     Qs = [torch.min(*(torch.einsum('sd, sd -> s', F, z) for F in Fs)) for Fs in [F_mean_s, F_rand_s]]
#                     self.actor_success = (Qs[0] > Qs[1]).cpu().numpy().tolist()
#             else:
#                 action = dist.sample()
#                 if step < self.num_expl_steps:
#                     action.uniform_(-1.0, 1.0)
#                 # if self.reward_free or infer:
#                 #     if step < self.num_expl_steps:
#                 #         action.uniform_(-1.0, 1.0)
#             return action.cpu().numpy()[0]
    
#     def aug_and_encode(self, obs):
#         obs = self.aug(obs)
#         return self.encoder(obs)

#     # 更新fb，包括保证FB满足度量损失、Fz满足Q损失和正则化损失三项
#     def update_fb(self, obs, action, discount, next_obs, next_goal, z, step):
#         metrics = {}
#         # compute target successor measure
#         with torch.no_grad():
#             if self.boltzmann:
#                 dist = self.actor(next_obs, z)
#                 next_action = dist.sample()
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(next_obs, z, stddev)
#                 next_action = dist.sample(clip=self.stddev_clip)
#             target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
#             target_B = self.backward_target_net(next_goal)  # batch x z_dim
#             target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
#             target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
#             target_M = torch.min(target_M1, target_M2)

#         # compute FB loss
#         F1, F2 = self.forward_net(obs, z, action)
#         B = self.backward_net(next_goal)
#         M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
#         M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
#         I = torch.eye(*M1.size(), device=M1.device)
#         off_diag = ~I.bool()
#         fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
#         fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
#         fb_loss = fb_offdiag + fb_diag

#         # Q LOSS
#         if self.q_loss:
#             with torch.no_grad():
#                 next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
#                 next_Q = torch.min(next_Q1, nextQ2)
#                 cov = torch.matmul(B.T, B) / B.shape[0]
#                 inv_cov = torch.inverse(cov)
#                 implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
#                 target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
#             Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
#             q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
#             fb_loss += self.q_loss_coef * q_loss

#         # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING
#         Cov = torch.matmul(B, B.T)
#         orth_loss_diag = - 2 * Cov.diag().mean()
#         orth_loss_offdiag = Cov[off_diag].pow(2).mean()
#         orth_loss = orth_loss_offdiag + orth_loss_diag
#         fb_loss += self.ortho_coef * orth_loss

#         # Cov = torch.cov(B.T)  # Vicreg loss
#         # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
#         # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
#         # orth_loss =  var_loss + cov_loss
#         # fb_loss += self.cfg.ortho_coef * orth_loss

#         if self.use_tb or self.use_wandb or self.use_hiplog:
#             metrics['target_M'] = target_M.mean().item()
#             metrics['M1'] = M1.mean().item()
#             metrics['F1'] = F1.mean().item()
#             metrics['B'] = B.mean().item()
#             metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
#             metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
#             metrics['fb_loss'] = fb_loss.item()
#             metrics['fb_diag'] = fb_diag.item()
#             metrics['fb_offdiag'] = fb_offdiag.item()
#             if self.q_loss:
#                 metrics['critic_target_q'] = target_Q.mean().item()
#                 metrics['critic_q1'] = Q1.mean().item()
#                 metrics['critic_q2'] = Q2.mean().item()
#                 metrics['critic_loss'] = q_loss.item()
#             metrics['orth_loss'] = orth_loss.item()
#             metrics['orth_loss_diag'] = orth_loss_diag.item()
#             metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
#             eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
#             metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
#             metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
#             if isinstance(self.fb_opt, torch.optim.Adam):
#                 metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

#         # optimize FB
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         self.fb_opt.zero_grad(set_to_none=True)
#         fb_loss.backward()
#         self.fb_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         return metrics

#     def update_critic(self, obs, action, z, reward, discount, next_obs, step):
#         metrics = dict()

#         with torch.no_grad():
#             if self.boltzmann:
#                 dist = self.actor(next_obs, z)
#                 next_action = dist.sample()
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(next_obs, z, stddev)
#                 next_action = dist.sample(clip=self.stddev_clip)
#             target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
#             target_Q1, target_Q2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
#             target_V = torch.min(target_Q1, target_Q2)
#             reward= reward.view(-1)
#             discount=discount.view(-1)
#             target_Q = reward + (discount * target_V)
            
#         F1, F2 = self.forward_net(obs, z, action)
#         Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
#         critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

#         if self.use_tb or self.use_wandb:
#             metrics['critic_target_q'] = target_Q.mean().item()
#             metrics['critic_q1'] = Q1.mean().item()
#             metrics['critic_q2'] = Q2.mean().item()
#             metrics['critic_loss'] = critic_loss.item()

#         # optimize critic
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         self.critic_opt.zero_grad(set_to_none=True)
#         critic_loss.backward()
#         self.critic_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         return metrics
    
#     def update_actor(self, obs, z, step):
#         metrics: tp.Dict[str, float] = {}
#         if self.boltzmann:
#             dist = self.actor(obs, z)
#             action = dist.rsample()
#         else:
#             stddev = utils.schedule(self.stddev_schedule, step)
#             dist = self.actor(obs, z, stddev)
#             action = dist.sample(clip=self.stddev_clip)

#         log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
#         F1, F2 = self.forward_net(obs, z, action)
#         Q1 = torch.einsum('sd, sd -> s', F1, z)
#         Q2 = torch.einsum('sd, sd -> s', F2, z)
#         if self.additional_metric:
#             q1_success = Q1 > Q2
#         Q = torch.min(Q1, Q2)

#         actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

#         # optimize actor
#         self.actor_opt.zero_grad(set_to_none=True)
#         actor_loss.backward()
#         self.actor_opt.step()

#         if self.use_tb or self.use_wandb:
#             if self.reward_free:
#                 metrics['actor_loss'] = actor_loss.item()
#                 metrics['q'] = Q.mean().item()
#                 if self.additional_metric:
#                     metrics['q1_success'] = q1_success.float().mean().item()
#                 metrics['actor_logprob'] = log_prob.mean().item()
#             else:
#                 metrics['actor_loss'] = actor_loss.item()
#                 metrics['actor_logprob'] = log_prob.mean().item()
#                 metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
#             # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

#         return metrics

#     def update(self, replay_iter, step):
#         metrics = {}

#         if self.reward_free:
#             self.rnd_agent.update(replay_iter, step)

#         if step % self.update_every_steps != 0:
#             return metrics

#         # 获取经验池
#         batch = next(replay_iter)
#         obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
#         next_goal = next_obs
#         if self.goal_space is not None:
#             assert batch.next_goal is not None
#             next_goal = batch.next_goal

#         if self.reward_free:
#             # 采样技能向量
#             z = self.sample_z(self.batch_size, device=self.device)
#             if not z.shape[-1] == self.z_dim:
#                 raise RuntimeError("There's something wrong with the logic here")

#             # 后向模型输入
#             backward_input = obs
#             future_goal = future_obs
#             if self.goal_space is not None:
#                 assert batch.goal is not None
#                 backward_input = goal
#                 future_goal = future_goal
#             perm = torch.randperm(self.batch_size)
#             backward_input = backward_input[perm]

#             # 产生z
#             if self.mix_ratio > 0:
#                 mix_idxs = np.where(np.random.uniform(size=self.batch_size) < self.mix_ratio)[0]
#                 if not self.rand_weight:
#                     with torch.no_grad():
#                         mix_z = self.backward_net(backward_input[mix_idxs]).detach()
#                 else:
#                     # generate random weight
#                     weight = torch.rand(size=(mix_idxs.shape[0], self.batch_size)).to(self.device)
#                     weight = F.normalize(weight, dim=1)
#                     uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.device)
#                     weight = uniform_rdv * weight
#                     with torch.no_grad():
#                         mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
#                 if self.norm_z:
#                     mix_z = math.sqrt(self.z_dim) * F.normalize(mix_z, dim=1)
#                 z[mix_idxs] = mix_z

#             # hindsight replay
#             if self.future_ratio > 0:
#                 assert future_goal is not None
#                 future_idxs = np.where(np.random.uniform(size=self.batch_size) < self.future_ratio)
#                 z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()

#             metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
#                                         next_obs=next_obs, next_goal=next_goal, z=z, step=step))
        
#         else:
#             z=skill
#             metrics.update(self.update_critic(obs=obs, action=action, discount=discount,
#                                         next_obs=next_obs, reward=reward, z=z, step=step))
        
#         # update actor
#         metrics.update(self.update_actor(obs, z, step))

#         # update critic target
#         utils.soft_update_params(self.forward_net, self.forward_target_net,
#                                  self.fb_target_tau)
#         utils.soft_update_params(self.backward_net, self.backward_target_net,
#                                  self.fb_target_tau)
#         utils.soft_update_params(self.critic, self.critic_target,
#                                  self.fb_target_tau)
        
#         return metrics

# # Fz+Q满足条件
# class fb6Agent:
#     def __init__(self, name, skill_dim, reward_free, obs_type, obs_shape, action_shape, device, lr, lr_coef, fb_target_tau, \
#                  update_every_steps, use_tb, use_wandb, num_expl_steps, num_inference_steps, hidden_dim, backward_hidden_dim, \
#                  feature_dim, z_dim, stddev_schedule, stddev_clip, update_z_every_step, update_z_proba, nstep, batch_size, init_fb, \
#                  update_encoder, goal_space, ortho_coef, log_std_bounds, temp, boltzmann, debug, future_ratio, mix_ratio, rand_weight, \
#                  preprocess, norm_z, q_loss, q_loss_coef, additional_metric, add_trunk):
#         # 参数
#         self.name = name
#         self.skill_dim = skill_dim
#         self.reward_free = reward_free
#         self.obs_type = obs_type
#         self.obs_shape = obs_shape
#         self.action_shape = action_shape
#         self.device = device
#         self.lr = lr
#         self.lr_coef = lr_coef
#         self.fb_target_tau = fb_target_tau
#         self.update_every_steps = update_every_steps
#         self.use_tb = use_tb
#         self.use_wandb = use_wandb
#         self.num_expl_steps = num_expl_steps
#         self.num_inference_steps = num_inference_steps
#         self.hidden_dim = hidden_dim
#         self.backward_hidden_dim = backward_hidden_dim
#         self.feature_dim = feature_dim
#         self.z_dim = z_dim
#         self.stddev_schedule = stddev_schedule
#         self.stddev_clip = stddev_clip
#         self.update_z_every_step = update_z_every_step
#         self.update_z_proba = update_z_proba
#         self.nstep = nstep
#         self.batch_size = batch_size
#         self.init_fb = init_fb
#         self.update_encoder = update_encoder
#         self.goal_space = goal_space
#         self.ortho_coef = ortho_coef
#         self.log_std_bounds = log_std_bounds
#         self.temp = temp
#         self.boltzmann = boltzmann
#         self.debug = debug
#         self.future_ratio = future_ratio
#         self.mix_ratio = mix_ratio
#         self.rand_weight = rand_weight
#         self.preprocess = preprocess
#         self.norm_z = norm_z
#         self.q_loss = q_loss
#         self.q_loss_coef = q_loss_coef
#         self.additional_metric = additional_metric
#         self.add_trunk = add_trunk
#         # 动作维度
#         self.action_dim = action_shape[0]
#         self.solved_meta = None
#         # models
#         if self.obs_type == 'pixels':
#             self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
#             self.encoder: nn.Module = Encoder(self.obs_shape).to(self.device)
#             self.obs_dim = self.encoder.repr_dim
#         else:
#             self.aug = nn.Identity()
#             self.encoder = nn.Identity()
#             self.obs_dim = self.obs_shape[0]
#         if self.feature_dim < self.obs_dim:
#             print(f"feature_dim {self.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
#         goal_dim = self.obs_dim
#         if self.goal_space is not None:
#             goal_dim = _goals.get_goal_space_dim(self.goal_space)
#         if self.z_dim < goal_dim:
#             print(f"z_dim {self.z_dim} should not be smaller that goal_dim {goal_dim}")
#         # create the network
#         if self.boltzmann:
#             self.actor: nn.Module = DiagGaussianActor(self.obs_dim, self.z_dim, self.action_dim,
#                                                       self.hidden_dim, self.log_std_bounds).to(self.device)
#         else:
#             self.actor = Actor(self.obs_dim, self.z_dim, self.action_dim, self.feature_dim, self.hidden_dim,
#                                preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # critic
#         self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
#                              feature_dim, hidden_dim).to(device)
#         self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
#                                     feature_dim, hidden_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         # forward
#         self.forward_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
#                                       self.feature_dim, self.hidden_dim,
#                                       preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # backward
#         if self.debug:
#             self.backward_net: nn.Module = IdentityMap().to(self.device)
#             self.backward_target_net: nn.Module = IdentityMap().to(self.device)
#         else:
#             self.backward_net = BackwardMap(goal_dim, self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
#             self.backward_target_net = BackwardMap(goal_dim,
#                                                    self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
#         # build up the target network
#         self.forward_target_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
#                                              self.feature_dim, self.hidden_dim,
#                                              preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # load the weights into the target networks
#         self.forward_target_net.load_state_dict(self.forward_net.state_dict())
#         self.backward_target_net.load_state_dict(self.backward_net.state_dict())
#         # optimizers
#         self.encoder_opt: tp.Optional[torch.optim.Adam] = None
#         if self.obs_type == 'pixels':
#             self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
#         self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
#         # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
#         # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
#         self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
#                                         {'params': self.backward_net.parameters(), 'lr': self.lr_coef * self.lr}],
#                                        lr=self.lr)

#         self.train()
#         self.critic_target.train()
#         self.forward_target_net.train()
#         self.backward_target_net.train()
#         self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
#         # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
#         # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
#         # self.online_cov.train()

#         self.rnd_agent = RNDAgent(rnd_rep_dim=512, update_encoder=update_encoder, rnd_scale=1., name='rnd', reward_free=reward_free, obs_type=obs_type, \
#                                   obs_shape=obs_shape, action_shape=action_shape, device=device, lr=lr, feature_dim=50, hidden_dim=hidden_dim, \
#                                   critic_target_tau=fb_target_tau, num_expl_steps=num_expl_steps, update_every_steps=update_every_steps, \
#                                   stddev_schedule=stddev_schedule, nstep=3, batch_size=batch_size, stddev_clip=stddev_clip, init_critic=init_fb, \
#                                   use_tb=use_tb, use_wandb=use_wandb)
        
        
#     # 加载模型
#     def init_from(self, other) -> None:
#         print("All attributes of 'other':", dir(other))
#         # copy parameters over
#         names = ["encoder", "actor"]
#         if self.init_fb:
#             names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
#         for name in names:
#             print("name", name)
#             utils.hard_update_params(getattr(other, name), getattr(self, name))
#         utils.hard_update_params(other.rnd_agent.actor, self.rnd_agent.actor)
#         # for key, val in self.__dict__.items():
#         #     if key not in ['critic_opt']:
#         #         if isinstance(val, torch.optim.Optimizer):
#         #             val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

#     def train(self, training=True):
#         self.training = training
#         for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
#             net.train(training)

#     def get_meta_specs(self):
#         return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

#     # 采样技能向量z
#     def sample_z(self, size, device="cpu"):
#         gaussian_rdv = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
#         gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * gaussian_rdv
#         else:
#             uniform_rdv = torch.rand((size, self.z_dim), dtype=torch.float32, device=device)
#             z = np.sqrt(self.z_dim) * uniform_rdv * gaussian_rdv
#         return z

#     # 初始化技能向量
#     def init_meta(self):
#         if self.solved_meta is not None:
#             return self.solved_meta
#         else:
#             z = self.sample_z(1)
#             z = z.squeeze().numpy()
#             meta = OrderedDict()
#             meta['skill'] = z
#         return meta

#     # 更新技能向量
#     def update_meta(self, meta, step, time_step):
#         if step % self.update_z_every_step == 0 and np.random.rand() < self.update_z_proba:
#             return self.init_meta()
#         return meta

#     # 根据goal点推理技能向量
#     def get_goal_meta(self, goal_array: np.ndarray):
#         desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             z = self.backward_net(desired_goal)
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
#         z = z.squeeze(0).cpu().numpy()
#         meta = OrderedDict()
#         meta['z'] = z
#         return meta

#     # 推断技能向量
#     def infer_meta(self, replay_iter, step):
#         obs_list, reward_list = [], []
#         batch_size = 0
#         while batch_size < self.num_inference_steps:
#             batch = next(replay_iter)
#             obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
#             next_goal = next_obs
#             obs_list.append(next_goal if self.goal_space is not None else next_obs)
#             reward_list.append(reward)
#             batch_size += next_obs.size(0)
#         obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
#         obs, reward = obs[:self.num_inference_steps], reward[:self.num_inference_steps]
#         return self.infer_meta_from_obs_and_rewards(obs, reward)

#     # 具体推断过程
#     def infer_meta_from_obs_and_rewards(self, obs, reward):
#         print('max reward: ', reward.max().cpu().item())
#         print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
#         print('median reward: ', reward.median().cpu().item())
#         print('min reward: ', reward.min().cpu().item())
#         print('mean reward: ', reward.mean().cpu().item())
#         print('num reward: ', reward.shape[0])

#         # filter out small reward
#         # pdb.set_trace()
#         # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
#         # obs = obs[idx]
#         # reward = reward[idx]
#         with torch.no_grad():
#             B = self.backward_net(obs)
#         z = torch.matmul(reward.T, B) / reward.shape[0]
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
#         meta = OrderedDict()
#         meta['skill'] = z.squeeze().cpu().numpy()
#         self.solved_meta = meta
#         return meta

#     # 执行动作
#     def act(self, obs, meta, step, eval_mode, infer=False):
#         if infer:
#             meta = OrderedDict()
#             action = self.rnd_agent.act(obs, meta, step+4000, eval_mode)
#             return action
#         if (not eval_mode and self.reward_free):
#             meta = OrderedDict()
#             action = self.rnd_agent.act(obs, meta, step, eval_mode)
#             return action
#         else:
#             obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
#             h = self.encoder(obs)
#             z = torch.as_tensor(meta['skill'], device=self.device).unsqueeze(0)  # type: ignore
#             if self.boltzmann:
#                 dist = self.actor(h, z)
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(h, z, stddev)
#             if eval_mode:
#                 action = dist.mean
#                 if self.additional_metric:
#                     # the following is doing extra computation only used for metrics,
#                     # it should be deactivated eventually
#                     F_mean_s = self.forward_net(obs, z, action)
#                     # F_samp_s = self.forward_net(obs, z, dist.sample())
#                     F_rand_s = self.forward_net(obs, z, torch.zeros_like(action).uniform_(-1.0, 1.0))
#                     Qs = [torch.min(*(torch.einsum('sd, sd -> s', F, z) for F in Fs)) for Fs in [F_mean_s, F_rand_s]]
#                     self.actor_success = (Qs[0] > Qs[1]).cpu().numpy().tolist()
#             else:
#                 action = dist.sample()
#                 if step < self.num_expl_steps:
#                     action.uniform_(-1.0, 1.0)
#                 # if self.reward_free or infer:
#                 #     if step < self.num_expl_steps:
#                 #         action.uniform_(-1.0, 1.0)
#             return action.cpu().numpy()[0]
    
#     def aug_and_encode(self, obs):
#         obs = self.aug(obs)
#         return self.encoder(obs)

#     # 更新fb，包括保证FB满足度量损失、Fz满足Q损失和正则化损失三项
#     def update_fb(self, obs, action, discount, next_obs, next_goal, z, step):
#         metrics = {}
#         # compute target successor measure
#         with torch.no_grad():
#             if self.boltzmann:
#                 dist = self.actor(next_obs, z)
#                 next_action = dist.sample()
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(next_obs, z, stddev)
#                 next_action = dist.sample(clip=self.stddev_clip)
#             target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
#             target_B = self.backward_target_net(next_goal)  # batch x z_dim
#             target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
#             target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
#             target_M = torch.min(target_M1, target_M2)

#         # compute FB loss
#         F1, F2 = self.forward_net(obs, z, action)
#         B = self.backward_net(next_goal)
#         M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
#         M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
#         I = torch.eye(*M1.size(), device=M1.device)
#         off_diag = ~I.bool()
#         fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
#         fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
#         fb_loss = fb_offdiag + fb_diag

#         # Q LOSS
#         if self.q_loss:
#             with torch.no_grad():
#                 next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
#                 next_Q = torch.min(next_Q1, nextQ2)
#                 cov = torch.matmul(B.T, B) / B.shape[0]
#                 inv_cov = torch.inverse(cov)
#                 implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
#                 target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
#             Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
#             q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
#             fb_loss += self.q_loss_coef * q_loss

#         # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING
#         Cov = torch.matmul(B, B.T)
#         orth_loss_diag = - 2 * Cov.diag().mean()
#         orth_loss_offdiag = Cov[off_diag].pow(2).mean()
#         orth_loss = orth_loss_offdiag + orth_loss_diag
#         fb_loss += self.ortho_coef * orth_loss

#         # Cov = torch.cov(B.T)  # Vicreg loss
#         # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
#         # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
#         # orth_loss =  var_loss + cov_loss
#         # fb_loss += self.cfg.ortho_coef * orth_loss

#         if self.use_tb or self.use_wandb or self.use_hiplog:
#             metrics['target_M'] = target_M.mean().item()
#             metrics['M1'] = M1.mean().item()
#             metrics['F1'] = F1.mean().item()
#             metrics['B'] = B.mean().item()
#             metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
#             metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
#             metrics['fb_loss'] = fb_loss.item()
#             metrics['fb_diag'] = fb_diag.item()
#             metrics['fb_offdiag'] = fb_offdiag.item()
#             if self.q_loss:
#                 metrics['critic_target_q'] = target_Q.mean().item()
#                 metrics['critic_q1'] = Q1.mean().item()
#                 metrics['critic_q2'] = Q2.mean().item()
#                 metrics['critic_loss'] = q_loss.item()
#             metrics['orth_loss'] = orth_loss.item()
#             metrics['orth_loss_diag'] = orth_loss_diag.item()
#             metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
#             eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
#             metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
#             metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
#             if isinstance(self.fb_opt, torch.optim.Adam):
#                 metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

#         # optimize FB
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         self.fb_opt.zero_grad(set_to_none=True)
#         fb_loss.backward()
#         self.fb_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         return metrics

#     def update_critic(self, obs, action, z, reward, discount, next_obs, step):
#         metrics = dict()

#         with torch.no_grad():
#             if self.boltzmann:
#                 dist = self.actor(next_obs, z)
#                 next_action = dist.sample()
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(next_obs, z, stddev)
#                 next_action = dist.sample(clip=self.stddev_clip)
#             target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
#             target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
#             target_MQ1, target_MQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
#             target_V = torch.min(target_Q1+target_MQ1, target_Q2+target_MQ2)
#             target_Q = reward + (discount * target_V)
            
#         F1, F2 = self.forward_net(obs, z, action)
#         MQ1, MQ2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]        
#         Q1, Q2 = self.critic(obs, action)
#         Q1, Q2 = Q1+MQ1, Q2+MQ2
#         critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

#         if self.use_tb or self.use_wandb:
#             metrics['critic_target_q'] = target_Q.mean().item()
#             metrics['critic_q1'] = Q1.mean().item()
#             metrics['critic_q2'] = Q2.mean().item()
#             metrics['critic_loss'] = critic_loss.item()

#         # optimize critic
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         self.critic_opt.zero_grad(set_to_none=True)
#         critic_loss.backward()
#         self.critic_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         return metrics
    
#     def update_actor(self, obs, z, step):
#         metrics: tp.Dict[str, float] = {}
#         if self.boltzmann:
#             dist = self.actor(obs, z)
#             action = dist.rsample()
#         else:
#             stddev = utils.schedule(self.stddev_schedule, step)
#             dist = self.actor(obs, z, stddev)
#             action = dist.sample(clip=self.stddev_clip)

#         log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
#         if self.reward_free:
#             F1, F2 = self.forward_net(obs, z, action)
#             Q1 = torch.einsum('sd, sd -> s', F1, z)
#             Q2 = torch.einsum('sd, sd -> s', F2, z)
#             if self.additional_metric:
#                 q1_success = Q1 > Q2
#             Q = torch.min(Q1, Q2)
#         else:
#             F1, F2 = self.forward_net(obs, z, action)
#             MQ1 = torch.einsum('sd, sd -> s', F1, z)
#             MQ2 = torch.einsum('sd, sd -> s', F2, z)
#             Q1, Q2 = self.critic(obs, action)
#             Q1, Q2 = Q1+MQ1, Q2+MQ2
#             # print("torch.min(Q1, Q2)", torch.min(Q1, Q2).mean())
#             # print("torch.min(MQ1, MQ2)", torch.min(MQ1, MQ2).mean())
#             Q = torch.min(Q1, Q2)
#             if self.use_tb or self.use_wandb:
#                 metrics['actor_MQ'] = torch.min(MQ1, MQ2).mean().item()
#                 metrics['actor_Q'] = torch.min(Q1, Q2).mean().item()
#         actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

#         # optimize actor
#         self.actor_opt.zero_grad(set_to_none=True)
#         actor_loss.backward()
#         self.actor_opt.step()

#         if self.use_tb or self.use_wandb:
#             if self.reward_free:
#                 metrics['actor_loss'] = actor_loss.item()
#                 metrics['q'] = Q.mean().item()
#                 if self.additional_metric:
#                     metrics['q1_success'] = q1_success.float().mean().item()
#                 metrics['actor_logprob'] = log_prob.mean().item()
#             else:
#                 metrics['actor_loss'] = actor_loss.item()
#                 metrics['actor_logprob'] = log_prob.mean().item()
#                 metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
#             # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

#         return metrics

#     def update(self, replay_iter, step):
#         metrics = {}

#         if self.reward_free:
#             self.rnd_agent.update(replay_iter, step)

#         if step % self.update_every_steps != 0:
#             return metrics

#         # 获取经验池
#         batch = next(replay_iter)
#         obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
#         next_goal = next_obs
#         if self.goal_space is not None:
#             assert batch.next_goal is not None
#             next_goal = batch.next_goal

#         if self.reward_free:
#             # 采样技能向量
#             z = self.sample_z(self.batch_size, device=self.device)
#             if not z.shape[-1] == self.z_dim:
#                 raise RuntimeError("There's something wrong with the logic here")

#             # 后向模型输入
#             backward_input = obs
#             future_goal = future_obs
#             if self.goal_space is not None:
#                 assert batch.goal is not None
#                 backward_input = goal
#                 future_goal = future_goal
#             perm = torch.randperm(self.batch_size)
#             backward_input = backward_input[perm]

#             # 产生z
#             if self.mix_ratio > 0:
#                 mix_idxs = np.where(np.random.uniform(size=self.batch_size) < self.mix_ratio)[0]
#                 if not self.rand_weight:
#                     with torch.no_grad():
#                         mix_z = self.backward_net(backward_input[mix_idxs]).detach()
#                 else:
#                     # generate random weight
#                     weight = torch.rand(size=(mix_idxs.shape[0], self.batch_size)).to(self.device)
#                     weight = F.normalize(weight, dim=1)
#                     uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.device)
#                     weight = uniform_rdv * weight
#                     with torch.no_grad():
#                         mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
#                 if self.norm_z:
#                     mix_z = math.sqrt(self.z_dim) * F.normalize(mix_z, dim=1)
#                 z[mix_idxs] = mix_z

#             # hindsight replay
#             if self.future_ratio > 0:
#                 assert future_goal is not None
#                 future_idxs = np.where(np.random.uniform(size=self.batch_size) < self.future_ratio)
#                 z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()

#             metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
#                                         next_obs=next_obs, next_goal=next_goal, z=z, step=step))
        
#         else:
#             z=skill
#             metrics.update(self.update_critic(obs=obs, action=action, discount=discount,
#                                         next_obs=next_obs, reward=reward, z=z, step=step))
        
#         # update actor
#         metrics.update(self.update_actor(obs, z, step))

#         # update critic target
#         utils.soft_update_params(self.forward_net, self.forward_target_net,
#                                  self.fb_target_tau)
#         utils.soft_update_params(self.backward_net, self.backward_target_net,
#                                  self.fb_target_tau)
#         utils.soft_update_params(self.critic, self.critic_target,
#                                  self.fb_target_tau)
        
#         return metrics

# # Fz+Q满足条件,取minQ
# class fb6Agent:
#     def __init__(self, name, skill_dim, reward_free, obs_type, obs_shape, action_shape, device, lr, lr_coef, fb_target_tau, \
#                  update_every_steps, use_tb, use_wandb, num_expl_steps, num_inference_steps, hidden_dim, backward_hidden_dim, \
#                  feature_dim, z_dim, stddev_schedule, stddev_clip, update_z_every_step, update_z_proba, nstep, batch_size, init_fb, \
#                  update_encoder, goal_space, ortho_coef, log_std_bounds, temp, boltzmann, debug, future_ratio, mix_ratio, rand_weight, \
#                  preprocess, norm_z, q_loss, q_loss_coef, additional_metric, add_trunk):
#         # 参数
#         self.name = name
#         self.skill_dim = skill_dim
#         self.reward_free = reward_free
#         self.obs_type = obs_type
#         self.obs_shape = obs_shape
#         self.action_shape = action_shape
#         self.device = device
#         self.lr = lr
#         self.lr_coef = lr_coef
#         self.fb_target_tau = fb_target_tau
#         self.update_every_steps = update_every_steps
#         self.use_tb = use_tb
#         self.use_wandb = use_wandb
#         self.num_expl_steps = num_expl_steps
#         self.num_inference_steps = num_inference_steps
#         self.hidden_dim = hidden_dim
#         self.backward_hidden_dim = backward_hidden_dim
#         self.feature_dim = feature_dim
#         self.z_dim = z_dim
#         self.stddev_schedule = stddev_schedule
#         self.stddev_clip = stddev_clip
#         self.update_z_every_step = update_z_every_step
#         self.update_z_proba = update_z_proba
#         self.nstep = nstep
#         self.batch_size = batch_size
#         self.init_fb = init_fb
#         self.update_encoder = update_encoder
#         self.goal_space = goal_space
#         self.ortho_coef = ortho_coef
#         self.log_std_bounds = log_std_bounds
#         self.temp = temp
#         self.boltzmann = boltzmann
#         self.debug = debug
#         self.future_ratio = future_ratio
#         self.mix_ratio = mix_ratio
#         self.rand_weight = rand_weight
#         self.preprocess = preprocess
#         self.norm_z = norm_z
#         self.q_loss = q_loss
#         self.q_loss_coef = q_loss_coef
#         self.additional_metric = additional_metric
#         self.add_trunk = add_trunk
#         # 动作维度
#         self.action_dim = action_shape[0]
#         self.solved_meta = None
#         # models
#         if self.obs_type == 'pixels':
#             self.aug: nn.Module = utils.RandomShiftsAug(pad=4)
#             self.encoder: nn.Module = Encoder(self.obs_shape).to(self.device)
#             self.obs_dim = self.encoder.repr_dim
#         else:
#             self.aug = nn.Identity()
#             self.encoder = nn.Identity()
#             self.obs_dim = self.obs_shape[0]
#         if self.feature_dim < self.obs_dim:
#             print(f"feature_dim {self.feature_dim} should not be smaller that obs_dim {self.obs_dim}")
#         goal_dim = self.obs_dim
#         if self.goal_space is not None:
#             goal_dim = _goals.get_goal_space_dim(self.goal_space)
#         if self.z_dim < goal_dim:
#             print(f"z_dim {self.z_dim} should not be smaller that goal_dim {goal_dim}")
#         # create the network
#         if self.boltzmann:
#             self.actor: nn.Module = DiagGaussianActor(self.obs_dim, self.z_dim, self.action_dim,
#                                                       self.hidden_dim, self.log_std_bounds).to(self.device)
#         else:
#             self.actor = Actor(self.obs_dim, self.z_dim, self.action_dim, self.feature_dim, self.hidden_dim,
#                                preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # critic
#         self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
#                              feature_dim, hidden_dim).to(device)
#         self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
#                                     feature_dim, hidden_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         # forward
#         self.forward_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
#                                       self.feature_dim, self.hidden_dim,
#                                       preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # backward
#         if self.debug:
#             self.backward_net: nn.Module = IdentityMap().to(self.device)
#             self.backward_target_net: nn.Module = IdentityMap().to(self.device)
#         else:
#             self.backward_net = BackwardMap(goal_dim, self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
#             self.backward_target_net = BackwardMap(goal_dim,
#                                                    self.z_dim, self.backward_hidden_dim, norm_z=self.norm_z).to(self.device)
#         # build up the target network
#         self.forward_target_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
#                                              self.feature_dim, self.hidden_dim,
#                                              preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
#         # load the weights into the target networks
#         self.forward_target_net.load_state_dict(self.forward_net.state_dict())
#         self.backward_target_net.load_state_dict(self.backward_net.state_dict())
#         # optimizers
#         self.encoder_opt: tp.Optional[torch.optim.Adam] = None
#         if self.obs_type == 'pixels':
#             self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
#         self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
#         # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
#         # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
#         self.fb_opt = torch.optim.Adam([{'params': self.forward_net.parameters()},  # type: ignore
#                                         {'params': self.backward_net.parameters(), 'lr': self.lr_coef * self.lr}],
#                                        lr=self.lr)

#         self.train()
#         self.critic_target.train()
#         self.forward_target_net.train()
#         self.backward_target_net.train()
#         self.actor_success: tp.List[float] = []  # only for debugging, can be removed eventually
#         # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
#         # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
#         # self.online_cov.train()

#         self.rnd_agent = RNDAgent(rnd_rep_dim=512, update_encoder=update_encoder, rnd_scale=1., name='rnd', reward_free=reward_free, obs_type=obs_type, \
#                                   obs_shape=obs_shape, action_shape=action_shape, device=device, lr=lr, feature_dim=50, hidden_dim=hidden_dim, \
#                                   critic_target_tau=fb_target_tau, num_expl_steps=num_expl_steps, update_every_steps=update_every_steps, \
#                                   stddev_schedule=stddev_schedule, nstep=3, batch_size=batch_size, stddev_clip=stddev_clip, init_critic=init_fb, \
#                                   use_tb=use_tb, use_wandb=use_wandb)
        
        
#     # 加载模型
#     def init_from(self, other) -> None:
#         print("All attributes of 'other':", dir(other))
#         # copy parameters over
#         names = ["encoder", "actor"]
#         if self.init_fb:
#             names += ["forward_net", "backward_net", "backward_target_net", "forward_target_net"]
#         for name in names:
#             print("name", name)
#             utils.hard_update_params(getattr(other, name), getattr(self, name))
#         utils.hard_update_params(other.rnd_agent.actor, self.rnd_agent.actor)
#         # for key, val in self.__dict__.items():
#         #     if key not in ['critic_opt']:
#         #         if isinstance(val, torch.optim.Optimizer):
#         #             val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

#     def train(self, training=True):
#         self.training = training
#         for net in [self.encoder, self.actor, self.forward_net, self.backward_net]:
#             net.train(training)

#     def get_meta_specs(self):
#         return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

#     # 采样技能向量z
#     def sample_z(self, size, device="cpu"):
#         gaussian_rdv = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
#         gaussian_rdv = F.normalize(gaussian_rdv, dim=1)
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * gaussian_rdv
#         else:
#             uniform_rdv = torch.rand((size, self.z_dim), dtype=torch.float32, device=device)
#             z = np.sqrt(self.z_dim) * uniform_rdv * gaussian_rdv
#         return z

#     # 初始化技能向量
#     def init_meta(self):
#         if self.solved_meta is not None:
#             return self.solved_meta
#         else:
#             z = self.sample_z(1)
#             z = z.squeeze().numpy()
#             meta = OrderedDict()
#             meta['skill'] = z
#         return meta

#     # 更新技能向量
#     def update_meta(self, meta, step, time_step):
#         if step % self.update_z_every_step == 0 and np.random.rand() < self.update_z_proba:
#             return self.init_meta()
#         return meta

#     # 根据goal点推理技能向量
#     def get_goal_meta(self, goal_array: np.ndarray):
#         desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             z = self.backward_net(desired_goal)
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
#         z = z.squeeze(0).cpu().numpy()
#         meta = OrderedDict()
#         meta['z'] = z
#         return meta

#     # 推断技能向量
#     def infer_meta(self, replay_iter, step):
#         obs_list, reward_list = [], []
#         batch_size = 0
#         while batch_size < self.num_inference_steps:
#             batch = next(replay_iter)
#             obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
#             next_goal = next_obs
#             obs_list.append(next_goal if self.goal_space is not None else next_obs)
#             reward_list.append(reward)
#             batch_size += next_obs.size(0)
#         obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
#         obs, reward = obs[:self.num_inference_steps], reward[:self.num_inference_steps]
#         return self.infer_meta_from_obs_and_rewards(obs, reward)

#     # 具体推断过程
#     def infer_meta_from_obs_and_rewards(self, obs, reward):
#         print('max reward: ', reward.max().cpu().item())
#         print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
#         print('median reward: ', reward.median().cpu().item())
#         print('min reward: ', reward.min().cpu().item())
#         print('mean reward: ', reward.mean().cpu().item())
#         print('num reward: ', reward.shape[0])

#         # filter out small reward
#         # pdb.set_trace()
#         # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
#         # obs = obs[idx]
#         # reward = reward[idx]
#         with torch.no_grad():
#             B = self.backward_net(obs)
#         z = torch.matmul(reward.T, B) / reward.shape[0]
#         if self.norm_z:
#             z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
#         meta = OrderedDict()
#         meta['skill'] = z.squeeze().cpu().numpy()
#         self.solved_meta = meta
#         return meta

#     # 执行动作
#     def act(self, obs, meta, step, eval_mode, infer=False):
#         if infer:
#             meta = OrderedDict()
#             action = self.rnd_agent.act(obs, meta, step+4000, eval_mode)
#             return action
#         if (not eval_mode and self.reward_free):
#             meta = OrderedDict()
#             action = self.rnd_agent.act(obs, meta, step, eval_mode)
#             return action
#         else:
#             obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
#             h = self.encoder(obs)
#             z = torch.as_tensor(meta['skill'], device=self.device).unsqueeze(0)  # type: ignore
#             if self.boltzmann:
#                 dist = self.actor(h, z)
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(h, z, stddev)
#             if eval_mode:
#                 action = dist.mean
#                 if self.additional_metric:
#                     # the following is doing extra computation only used for metrics,
#                     # it should be deactivated eventually
#                     F_mean_s = self.forward_net(obs, z, action)
#                     # F_samp_s = self.forward_net(obs, z, dist.sample())
#                     F_rand_s = self.forward_net(obs, z, torch.zeros_like(action).uniform_(-1.0, 1.0))
#                     Qs = [torch.min(*(torch.einsum('sd, sd -> s', F, z) for F in Fs)) for Fs in [F_mean_s, F_rand_s]]
#                     self.actor_success = (Qs[0] > Qs[1]).cpu().numpy().tolist()
#             else:
#                 action = dist.sample()
#                 if step < self.num_expl_steps:
#                     action.uniform_(-1.0, 1.0)
#                 # if self.reward_free or infer:
#                 #     if step < self.num_expl_steps:
#                 #         action.uniform_(-1.0, 1.0)
#             return action.cpu().numpy()[0]
    
#     def aug_and_encode(self, obs):
#         obs = self.aug(obs)
#         return self.encoder(obs)

#     # 更新fb，包括保证FB满足度量损失、Fz满足Q损失和正则化损失三项
#     def update_fb(self, obs, action, discount, next_obs, next_goal, z, step):
#         metrics = {}
#         # compute target successor measure
#         with torch.no_grad():
#             if self.boltzmann:
#                 dist = self.actor(next_obs, z)
#                 next_action = dist.sample()
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(next_obs, z, stddev)
#                 next_action = dist.sample(clip=self.stddev_clip)
#             target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
#             target_B = self.backward_target_net(next_goal)  # batch x z_dim
#             target_M1 = torch.einsum('sd, td -> st', target_F1, target_B)  # batch x batch
#             target_M2 = torch.einsum('sd, td -> st', target_F2, target_B)  # batch x batch
#             target_M = torch.min(target_M1, target_M2)

#         # compute FB loss
#         F1, F2 = self.forward_net(obs, z, action)
#         B = self.backward_net(next_goal)
#         M1 = torch.einsum('sd, td -> st', F1, B)  # batch x batch
#         M2 = torch.einsum('sd, td -> st', F2, B)  # batch x batch
#         I = torch.eye(*M1.size(), device=M1.device)
#         off_diag = ~I.bool()
#         fb_offdiag: tp.Any = 0.5 * sum((M - discount * target_M)[off_diag].pow(2).mean() for M in [M1, M2])
#         fb_diag: tp.Any = -sum(M.diag().mean() for M in [M1, M2])
#         fb_loss = fb_offdiag + fb_diag

#         # Q LOSS
#         if self.q_loss:
#             with torch.no_grad():
#                 next_Q1, nextQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
#                 next_Q = torch.min(next_Q1, nextQ2)
#                 cov = torch.matmul(B.T, B) / B.shape[0]
#                 inv_cov = torch.inverse(cov)
#                 implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=1)  # batch_size
#                 target_Q = implicit_reward.detach() + discount.squeeze(1) * next_Q  # batch_size
#             Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
#             q_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
#             fb_loss += self.q_loss_coef * q_loss

#         # ORTHONORMALITY LOSS FOR BACKWARD EMBEDDING
#         Cov = torch.matmul(B, B.T)
#         orth_loss_diag = - 2 * Cov.diag().mean()
#         orth_loss_offdiag = Cov[off_diag].pow(2).mean()
#         orth_loss = orth_loss_offdiag + orth_loss_diag
#         fb_loss += self.ortho_coef * orth_loss

#         # Cov = torch.cov(B.T)  # Vicreg loss
#         # var_loss = F.relu(1 - Cov.diag().clamp(1e-4, 1).sqrt()).mean()  # eps avoids inf. sqrt gradient at 0
#         # cov_loss = 2 * torch.triu(Cov, diagonal=1).pow(2).mean() # 2x upper triangular part
#         # orth_loss =  var_loss + cov_loss
#         # fb_loss += self.cfg.ortho_coef * orth_loss

#         if self.use_tb or self.use_wandb or self.use_hiplog:
#             metrics['target_M'] = target_M.mean().item()
#             metrics['M1'] = M1.mean().item()
#             metrics['F1'] = F1.mean().item()
#             metrics['B'] = B.mean().item()
#             metrics['B_norm'] = torch.norm(B, dim=-1).mean().item()
#             metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
#             metrics['fb_loss'] = fb_loss.item()
#             metrics['fb_diag'] = fb_diag.item()
#             metrics['fb_offdiag'] = fb_offdiag.item()
#             if self.q_loss:
#                 metrics['critic_target_q'] = target_Q.mean().item()
#                 metrics['critic_q1'] = Q1.mean().item()
#                 metrics['critic_q2'] = Q2.mean().item()
#                 metrics['critic_loss'] = q_loss.item()
#             metrics['orth_loss'] = orth_loss.item()
#             metrics['orth_loss_diag'] = orth_loss_diag.item()
#             metrics['orth_loss_offdiag'] = orth_loss_offdiag.item()
#             eye_diff = torch.matmul(B.T, B) / B.shape[0] - torch.eye(B.shape[1], device=B.device)
#             metrics['orth_linf'] = torch.max(torch.abs(eye_diff)).item()
#             metrics['orth_l2'] = eye_diff.norm().item() / math.sqrt(B.shape[1])
#             if isinstance(self.fb_opt, torch.optim.Adam):
#                 metrics["fb_opt_lr"] = self.fb_opt.param_groups[0]["lr"]

#         # optimize FB
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         self.fb_opt.zero_grad(set_to_none=True)
#         fb_loss.backward()
#         self.fb_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         return metrics

#     def update_critic(self, obs, action, z, reward, discount, next_obs, step):
#         metrics = dict()

#         with torch.no_grad():
#             if self.boltzmann:
#                 dist = self.actor(next_obs, z)
#                 next_action = dist.sample()
#             else:
#                 stddev = utils.schedule(self.stddev_schedule, step)
#                 dist = self.actor(next_obs, z, stddev)
#                 next_action = dist.sample(clip=self.stddev_clip)
#             target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
#             target_F1, target_F2 = self.forward_target_net(next_obs, z, next_action)  # batch x z_dim
#             target_MQ1, target_MQ2 = [torch.einsum('sd, sd -> s', target_Fi, z) for target_Fi in [target_F1, target_F2]]
#             target_V = torch.min(target_Q1, target_Q2) + torch.min(target_MQ1, target_MQ2)
#             target_Q = reward + (discount * target_V)
            
#         F1, F2 = self.forward_net(obs, z, action)
#         MQ1, MQ2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]     
#         MQ=torch.min(MQ1,MQ2)
#         Q1, Q2 = self.critic(obs, action)
#         Q1, Q2 = Q1+MQ, Q2+MQ
#         critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

#         if self.use_tb or self.use_wandb:
#             metrics['critic_target_q'] = target_Q.mean().item()
#             metrics['critic_q1'] = Q1.mean().item()
#             metrics['critic_q2'] = Q2.mean().item()
#             metrics['critic_loss'] = critic_loss.item()

#         # optimize critic
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         self.critic_opt.zero_grad(set_to_none=True)
#         critic_loss.backward()
#         self.critic_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         return metrics
    
#     def update_actor(self, obs, z, step):
#         metrics: tp.Dict[str, float] = {}
#         if self.boltzmann:
#             dist = self.actor(obs, z)
#             action = dist.rsample()
#         else:
#             stddev = utils.schedule(self.stddev_schedule, step)
#             dist = self.actor(obs, z, stddev)
#             action = dist.sample(clip=self.stddev_clip)

#         log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
#         if self.reward_free:
#             F1, F2 = self.forward_net(obs, z, action)
#             Q1 = torch.einsum('sd, sd -> s', F1, z)
#             Q2 = torch.einsum('sd, sd -> s', F2, z)
#             if self.additional_metric:
#                 q1_success = Q1 > Q2
#             Q = torch.min(Q1, Q2)
#         else:
#             F1, F2 = self.forward_net(obs, z, action)
#             MQ1 = torch.einsum('sd, sd -> s', F1, z)
#             MQ2 = torch.einsum('sd, sd -> s', F2, z)
#             Q1, Q2 = self.critic(obs, action)
#             Q1, Q2 = Q1+MQ1, Q2+MQ2
#             # print("torch.min(Q1, Q2)", torch.min(Q1, Q2).mean())
#             # print("torch.min(MQ1, MQ2)", torch.min(MQ1, MQ2).mean())
#             Q = torch.min(Q1, Q2)
#             if self.use_tb or self.use_wandb:
#                 metrics['actor_MQ'] = torch.min(MQ1, MQ2).mean().item()
#                 metrics['actor_Q'] = torch.min(Q1, Q2).mean().item()
#         actor_loss = (self.temp * log_prob - Q).mean() if self.boltzmann else -Q.mean()

#         # optimize actor
#         self.actor_opt.zero_grad(set_to_none=True)
#         actor_loss.backward()
#         self.actor_opt.step()

#         if self.use_tb or self.use_wandb:
#             if self.reward_free:
#                 metrics['actor_loss'] = actor_loss.item()
#                 metrics['q'] = Q.mean().item()
#                 if self.additional_metric:
#                     metrics['q1_success'] = q1_success.float().mean().item()
#                 metrics['actor_logprob'] = log_prob.mean().item()
#             else:
#                 metrics['actor_loss'] = actor_loss.item()
#                 metrics['actor_logprob'] = log_prob.mean().item()
#                 metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
#             # metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

#         return metrics

#     def update(self, replay_iter, step):
#         metrics = {}

#         if self.reward_free:
#             self.rnd_agent.update(replay_iter, step)

#         if step % self.update_every_steps != 0:
#             return metrics

#         # 获取经验池
#         batch = next(replay_iter)
#         obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
#         next_goal = next_obs
#         if self.goal_space is not None:
#             assert batch.next_goal is not None
#             next_goal = batch.next_goal

#         if self.reward_free:
#             # 采样技能向量
#             z = self.sample_z(self.batch_size, device=self.device)
#             if not z.shape[-1] == self.z_dim:
#                 raise RuntimeError("There's something wrong with the logic here")

#             # 后向模型输入
#             backward_input = obs
#             future_goal = future_obs
#             if self.goal_space is not None:
#                 assert batch.goal is not None
#                 backward_input = goal
#                 future_goal = future_goal
#             perm = torch.randperm(self.batch_size)
#             backward_input = backward_input[perm]

#             # 产生z
#             if self.mix_ratio > 0:
#                 mix_idxs = np.where(np.random.uniform(size=self.batch_size) < self.mix_ratio)[0]
#                 if not self.rand_weight:
#                     with torch.no_grad():
#                         mix_z = self.backward_net(backward_input[mix_idxs]).detach()
#                 else:
#                     # generate random weight
#                     weight = torch.rand(size=(mix_idxs.shape[0], self.batch_size)).to(self.device)
#                     weight = F.normalize(weight, dim=1)
#                     uniform_rdv = torch.rand(mix_idxs.shape[0], 1).to(self.device)
#                     weight = uniform_rdv * weight
#                     with torch.no_grad():
#                         mix_z = torch.matmul(weight, self.backward_net(backward_input).detach())
#                 if self.norm_z:
#                     mix_z = math.sqrt(self.z_dim) * F.normalize(mix_z, dim=1)
#                 z[mix_idxs] = mix_z

#             # hindsight replay
#             if self.future_ratio > 0:
#                 assert future_goal is not None
#                 future_idxs = np.where(np.random.uniform(size=self.batch_size) < self.future_ratio)
#                 z[future_idxs] = self.backward_net(future_goal[future_idxs]).detach()

#             metrics.update(self.update_fb(obs=obs, action=action, discount=discount,
#                                         next_obs=next_obs, next_goal=next_goal, z=z, step=step))
        
#         else:
#             z=skill
#             metrics.update(self.update_critic(obs=obs, action=action, discount=discount,
#                                         next_obs=next_obs, reward=reward, z=z, step=step))
        
#         # update actor
#         metrics.update(self.update_actor(obs, z, step))

#         # update critic target
#         utils.soft_update_params(self.forward_net, self.forward_target_net,
#                                  self.fb_target_tau)
#         utils.soft_update_params(self.backward_net, self.backward_target_net,
#                                  self.fb_target_tau)
#         utils.soft_update_params(self.critic, self.critic_target,
#                                  self.fb_target_tau)
        
#         return metrics