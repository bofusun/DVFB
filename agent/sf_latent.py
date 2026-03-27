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
from agent.ddpg import Encoder
from agent.ddpg import DDPGAgent
from agent.fb_modules import IdentityMap
from agent.fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, mlp, OnlineCov
from agent.ddpg import Critic, Critic_C
        
class FeatureLearner(nn.Module):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net: nn.Module = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        return None

class Identity(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.feature_net = nn.Identity()
        
class Laplacian(FeatureLearner):
    def forward(self, obs, action, next_obs, future_obs):
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        loss = (phi - next_phi).pow(2).mean()
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss

class ContrastiveFeature(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)
        # self.W = nn.Linear(z_dim, z_dim, bias=False)
        # nn.init.orthogonal_(self.W.weight.data, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del next_obs
        assert future_obs is not None
        # phi = self.feature_net(obs)
        # future_phi = self.feature_net(future_obs)
        # phi = F.normalize(phi, dim=1)
        # future_phi = F.normalize(future_phi, dim=1)
        phi = self.feature_net(obs)
        future_mu = self.mu_net(future_obs)
        phi = F.normalize(phi, dim=1)
        future_mu = F.normalize(future_mu, dim=1)
        logits = torch.einsum('sd, td-> st', phi, future_mu)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = - logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss

        # loss = - logits.diag().mean() + 0.5 * logits[off_diag].pow(2).mean()

        # orthonormality loss
        # Cov = torch.matmul(phi, phi.T)
        # I = torch.eye(*Cov.size(), device=Cov.device)
        # off_diag = ~I.bool()
        # orth_loss_diag = - 2 * Cov.diag().mean()
        # orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        # orth_loss = orth_loss_offdiag + orth_loss_diag
        # loss += orth_loss
        # normalize to compute cosine distance
        # phi = F.normalize(phi, dim=1)
        # future_phi = F.normalize(future_phi, dim=1)
        # logits = torch.einsum('sd, td-> st', phi, future_phi) # batch x batch
        # labels = torch.eye(*logits.size(), out=torch.empty_like(logits))
        # # - labels * torch.log(torch.sigmoid(logits)) - (1 - labels) * torch.log(1 - torch.sigmoid(logits))
        # loss = F.binary_cross_entropy(torch.sigmoid(logits), labels)


class ContrastiveFeaturev2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)
        # self.W = nn.Linear(z_dim, z_dim, bias=False)
        # nn.init.orthogonal_(self.W.weight.data, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del action
        del next_obs
        assert future_obs is not None
        # phi = self.feature_net(obs)
        # future_phi = self.feature_net(future_obs)
        # phi = F.normalize(phi, dim=1)
        # future_phi = F.normalize(future_phi, dim=1)
        future_phi = self.feature_net(future_obs)
        mu = self.mu_net(obs)
        future_phi = F.normalize(future_phi, dim=1)
        mu = F.normalize(mu, dim=1)
        logits = torch.einsum('sd, td-> st', mu, future_phi)  # batch x batch
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = - logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ICM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        # self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.inverse_dynamic_net = mlp(2 * z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', action_dim, 'tanh')
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        # predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        # forward_error = (next_phi.detach() - predicted_next_obs).pow(2).mean()
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        icm_loss = backward_error
        # icm_loss = forward_error + backward_error
        return icm_loss


class TransitionModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_obs).pow(2).mean()
        return forward_error


class TransitionLatentModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.forward_dynamic_net = mlp(z_dim + action_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        with torch.no_grad():
            next_phi = self.target_feature_net(next_obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_phi.detach()).pow(2).mean()
        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)

        return forward_error


class AutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.decoder = mlp(z_dim, hidden_dim, 'irelu', hidden_dim, 'irelu', obs_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        del next_obs
        del action
        phi = self.feature_net(obs)
        predicted_obs = self.decoder(phi)
        reconstruction_error = (predicted_obs - obs).pow(2).mean()
        return reconstruction_error


class SVDSR(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.target_mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(obs)
        mu = self.mu_net(next_obs)
        SR = torch.einsum("sd, td -> st", phi, mu)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_phi, target_mu)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = - 2 * SR.diag().mean() + (SR - 0.99 * target_SR.detach())[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utils.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDSRv2(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.target_feature_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.target_mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(obs)
        SR = torch.einsum("sd, td -> st", mu, phi)
        with torch.no_grad():
            target_phi = self.target_feature_net(next_obs)
            target_mu = self.target_mu_net(next_obs)
            target_SR = torch.einsum("sd, td -> st", target_mu, target_phi)

        I = torch.eye(*SR.size(), device=SR.device)
        off_diag = ~I.bool()
        loss = - 2 * SR.diag().mean() + (SR - 0.98 * target_SR.detach())[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        utils.soft_update_params(self.feature_net, self.target_feature_net, 0.01)
        utils.soft_update_params(self.mu_net, self.target_mu_net, 0.01)

        return loss


class SVDP(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim + action_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor, future_obs: torch.Tensor):
        del future_obs
        phi = self.feature_net(next_obs)
        mu = self.mu_net(torch.cat([obs, action], dim=1))
        P = torch.einsum("sd, td -> st", mu, phi)
        I = torch.eye(*P.size(), device=P.device)
        off_diag = ~I.bool()
        loss = - 2 * P.diag().mean() + P[off_diag].pow(2).mean()

        # orthonormality loss
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = - 2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        return loss
    
# 加一个普通Q
class SFLATENTAgent:
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
        self.successor_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
                                      self.feature_dim, self.hidden_dim,
                                      preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
        # build up the target network
        self.successor_target_net = ForwardMap(self.obs_dim, self.z_dim, self.action_dim,
                                      self.feature_dim, self.hidden_dim,
                                      preprocess=self.preprocess, add_trunk=self.add_trunk).to(self.device)
        learner = dict(icm=ICM, transition=TransitionModel, latent=TransitionLatentModel,
                       contrastive=ContrastiveFeature, autoencoder=AutoEncoder, lap=Laplacian,
                       random=FeatureLearner, svd_sr=SVDSR, svd_p=SVDP,
                       contrastivev2=ContrastiveFeaturev2, svd_srv2=SVDSRv2,
                       identity=Identity)['latent']
        self.feature_learner = learner(goal_dim, self.action_dim, self.z_dim, self.backward_hidden_dim).to(self.device)
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())
        # optimizers
        self.encoder_opt: tp.Optional[torch.optim.Adam] = None
        if self.obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        # params = [p for net in [self.forward_net, self.backward_net] for p in net.parameters()]
        # self.fb_opt = torch.optim.Adam(params, lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=lr)
        self.phi_opt = torch.optim.Adam(self.feature_learner.parameters(), lr=self.lr_coef * self.lr)
            
        self.train()
        self.critic_target.train()
        self.successor_target_net.train()
        self.inv_cov = torch.eye(self.z_dim, dtype=torch.float32, device=self.device)
        self.actor_success = []  # only for debugging, can be removed eventually
        # self.inv_cov = torch.eye(self.cfg.z_dim, dtype=torch.float32, device=self.cfg.device)
        # self.online_cov = OnlineCov(mom=0.99, dim=self.cfg.z_dim).to(self.cfg.device)
        # self.online_cov.train()

    # 加载模型
    def init_from(self, other) -> None:
        # copy parameters over
        names = ["encoder", "actor"]
        if self.init_fb:
            names += ["successor_net", "feature_learner", "successor_target_net"]
        for name in names:
            print("name", name)
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        # for key, val in self.__dict__.items():
        #     if isinstance(val, torch.optim.Optimizer):
        #         val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.encoder, self.actor, self.successor_net]:
            net.train(training)
        if self.phi_opt is not None:
            self.feature_learner.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    # 采样技能向量z
    def sample_z(self, size, device="cpu"):
        gaussian_rdv = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        z = math.sqrt(self.z_dim) * F.normalize(gaussian_rdv, dim=1)
        return z

    # 初始化技能向量
    def init_meta(self):
        if self.solved_meta is not None:
            # print('solved_meta', self.solved_meta)
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

    def get_goal_meta(self, goal_array: np.ndarray):
        # assert self.cfg.feature_learner in ["FB"]

        desired_goal = torch.tensor(goal_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z = self.feature_learner.feature_net(desired_goal)

        z = torch.matmul(z, self.inv_cov)  # 1 x z_dim
        z = math.sqrt(self.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta['skill'] = z
        return meta
    
    # 推断技能向量
    def infer_meta(self, replay_iter, step):
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < self.num_inference_steps:
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
            next_goal = next_obs
            obs_list.append(next_goal if self.goal_space is not None else next_obs)
            reward_list.append(reward)
            batch_size += next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[:self.num_inference_steps], reward[:self.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, reward)

    # 具体推断过程
    def infer_meta_from_obs_and_rewards(self, obs, reward):
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        # filter out small reward
        # pdb.set_trace()
        # idx = torch.where(reward >= torch.quantile(reward, 0.99))[0]
        # obs = obs[idx]
        # reward = reward[idx]
        with torch.no_grad():
            obs = self.encoder(obs)
            phi = self.feature_learner.feature_net(obs)
        z = torch.linalg.lstsq(phi, reward).solution  # z_dim x 1
        z = math.sqrt(self.z_dim) * F.normalize(z, dim=0)  # be careful to the dimension
        meta = OrderedDict()
        meta['skill'] = z.squeeze().cpu().numpy()
        self.solved_meta = meta
        return meta

    # 执行动作
    def act(self, obs, meta, step, eval_mode, infer=False):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)  # type: ignore
        h = self.encoder(obs)
        z = torch.as_tensor(meta['skill'], device=self.device).unsqueeze(0)  # type: ignore
        if self.boltzmann:
            dist = self.actor(h, z)
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(h, z, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            # if step < self.num_expl_steps:
            #     action.uniform_(-1.0, 1.0)
            if self.reward_free or infer:
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    # 更新fb，包括保证FB满足度量损失、Fz满足Q损失和正则化损失三项
    def update_sf(self, obs, action, discount, next_obs, next_goal, z, step):
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
            next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)  # batch x z_dim
            target_phi = self.feature_learner.feature_net(next_goal).detach()  # batch x z_dim
            next_Q1, next_Q2 = [torch.einsum('sd, sd -> s', next_Fi, z) for next_Fi in [next_F1, next_F2]]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discount * next_F

        # compute FB loss
        F1, F2 = self.successor_net(obs, z, action)
        if not self.q_loss:
            # compute SF loss
            sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)
        else:
            # alternative loss
            Q1, Q2 = [torch.einsum('sd, sd -> s', Fi, z) for Fi in [F1, F2]]
            target_Q = torch.einsum('sd, sd -> s', target_F, z)
            sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            # sf_loss /= self.cfg.z_dim

        # compute feature loss
        phi_loss = self.feature_learner(obs=obs, action=action, next_obs=next_goal, future_obs=next_goal)

        if self.use_tb or self.use_wandb:
            metrics['target_F'] = target_F.mean().item()
            metrics['F1'] = F1.mean().item()
            metrics['phi'] = target_phi.mean().item()
            metrics['phi_norm'] = torch.norm(target_phi, dim=-1).mean().item()
            metrics['z_norm'] = torch.norm(z, dim=-1).mean().item()
            metrics['sf_loss'] = sf_loss.item()
            if phi_loss is not None:
                metrics['phi_loss'] = phi_loss.item()
            if isinstance(self.sf_opt, torch.optim.Adam):
                metrics["sf_opt_lr"] = self.sf_opt.param_groups[0]["lr"]

        # optimize SF
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.sf_opt.zero_grad(set_to_none=True)
        if self.phi_opt is not None:
            self.phi_opt.zero_grad(set_to_none=True)
            phi_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.sf_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        if self.phi_opt is not None:
            self.phi_opt.step()
        return metrics

    def update_critic(self, obs, action, z, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            if self.boltzmann:
                dist = self.actor(next_obs, z)
                next_action = dist.sample()
            else:
                stddev = utils.schedule(self.stddev_schedule, step)
                dist = self.actor(next_obs, z, stddev)
                next_action = dist.sample(clip=self.stddev_clip)
            obs = torch.cat([obs, z], dim=1)
            next_obs = torch.cat([next_obs, z], dim=1)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

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
            F1, F2 = self.successor_net(obs, z, action)
            Q1 = torch.einsum('sd, sd -> s', F1, z)
            Q2 = torch.einsum('sd, sd -> s', F2, z)
            if self.additional_metric:
                q1_success = Q1 > Q2
        else:
            Q1, Q2 = self.critic(torch.cat([obs, z], dim=1), action)
        Q = torch.min(Q1, Q2)
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

    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        # 获取经验池
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, future_obs, skill = utils.to_torch(batch, self.device)
        # 视觉编码 augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            future_obs = self.aug_and_encode(future_obs)
        
        next_goal = next_obs
        if self.goal_space is not None:
            assert batch.next_goal is not None
            next_goal = batch.next_goal

        if self.reward_free:
            # 采样技能向量
            z = self.sample_z(self.batch_size, device=self.device)
            if not z.shape[-1] == self.z_dim:
                raise RuntimeError("There's something wrong with the logic here")

            # 产生z
            if self.mix_ratio > 0:
                perm = torch.randperm(self.batch_size)
                desired_goal = next_goal[perm]
                with torch.no_grad():
                    phi = self.feature_learner.feature_net(desired_goal)
                # compute inverse of cov of phi
                cov = torch.matmul(phi.T, phi) / phi.shape[0]
                inv_cov = torch.linalg.pinv(cov)

                mix_idxs = np.where(np.random.uniform(size=self.batch_size) < self.mix_ratio)[0]
                with torch.no_grad():
                    new_z = phi[mix_idxs]

                new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
                new_z = math.sqrt(self.z_dim) * F.normalize(new_z, dim=1)
                z[mix_idxs] = new_z

            metrics.update(self.update_sf(obs=obs, action=action, discount=discount,
                                        next_obs=next_obs, next_goal=next_goal, z=z, step=step))
        else:
            z=skill
            metrics.update(self.update_critic(obs=obs, action=action, discount=discount,
                                        next_obs=next_obs, reward=reward, z=z, step=step))
        # update actor
        metrics.update(self.update_actor(obs.detach(), z.detach(), step))

        # update critic target
        utils.soft_update_params(self.successor_net, self.successor_target_net,
                                     self.fb_target_tau)
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.fb_target_tau)
        
        return metrics