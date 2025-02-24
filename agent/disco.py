import math
from collections import OrderedDict, deque

import ipdb
import os
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
from sklearn.neighbors import KernelDensity, NearestNeighbors

import utils
from agent.sac import SACAgent

min_max_maze = {'square_bottleneck': [-0.5, 9.5, -0.5, 9.5], 
                'square_a': [-0.5, 4.5, -4.5, 0.5], 
                'square_b': [-0.5, 4.5, -4.5, 0.5],
                'square_c': [-0.5, 4.5, -4.5, 0.5], 
                'square_empty': [-0.5, 9.5, -10.5, -0.5]}

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)
    return f_x


class Discriminator(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, obs):
        skill_pred = self.skill_pred_net(obs)
        return skill_pred


class DISCOAgent(SACAgent):

    def __init__(self, update_skill_every_step, skill_dim, diayn_scale, update_encoder, max_skill_dim, discriminator_lr, eval_num_skills, dmc_update_D_delta, threshold,
                delta_threshold, extend_num, num_eval_each_skill, num_neigh, use_clamp, interval, kl_start, kl_target, check_delta_every_steps, init_beta, use_actor_target, only_x, **kwargs):
        self.skill_dim = skill_dim
        self.max_skill_dim = max_skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.update_encoder = update_encoder
        self.discriminator_lr = discriminator_lr
        self.eval_num_skills = eval_num_skills
        self.delta_threshold = delta_threshold
        self.extend_num = extend_num
        self.num_eval_each_skill = num_eval_each_skill
        self.num_neigh = num_neigh
        self.use_clamp = use_clamp
        self.use_actor_target = use_actor_target
        self.interval = interval
        self.kl_start = kl_start
        self.kl_target = kl_target
        self.check_delta_every_steps = check_delta_every_steps
        self.default_delta = self.delta_threshold
        self.dmc_update_D_delta = dmc_update_D_delta
        self.threshold = threshold

        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.max_skill_dim
        self.skill_type = 'discrete'

        # create actor and critic
        super().__init__(**kwargs)

        # to find guide skill
        self.D_delta_all = torch.ones(self.max_skill_dim, 1).to(kwargs['device']) # For D_delta target
        self.guide_skill = None
        self.delta_dict = {skill_idx : deque([torch.tensor(1.0).to(kwargs['device'])] * 5, maxlen=5) for skill_idx in range(self.max_skill_dim)}
        self.clamp_kl = self.kl_start
        self.log_beta = torch.tensor(np.log(init_beta)).to(kwargs['device'])
        self.count = self.skill_dim
        self.extended_step = 0
        self.save_snapshot = False
        self.min_max = min_max_maze[self.maze_type]
        
        # create discriminator
        self.discriminator = Discriminator(self.obs_dim - self.max_skill_dim, self.max_skill_dim,
                                               kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.discriminator_criterion = nn.CrossEntropyLoss()
        # optimizer
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr)

        self.discriminator.train()
        # dtype
        if self.dtype == 'float64':
            self.discriminator.to(torch.float64)
            self.log_beta.to(torch.float64)
        elif self.dtype == 'float32':
            self.discriminator.to(torch.float32)
            self.log_beta.to(torch.float32)

    @property
    def beta(self):
        return self.log_beta.exp()    

    def get_meta_specs(self):
        return (specs.Array((self.max_skill_dim,), np.float32, 'skill'),)

    def init_meta(self, weight=None):
        skill = np.zeros(self.max_skill_dim, dtype=self.dtype)
        # weighted sampling
        if weight is None:
            idx = np.random.choice(self.skill_dim)
        else:
            p = []
            for sdx in range(self.skill_dim):
                if len(weight[sdx]) == 0:
                    p.append(1.0)
                else:
                    p.append((sum(weight[sdx])/len(weight[sdx])).cpu().item())
            p = softmax(np.array(p))
            idx = np.random.choice(self.skill_dim, p = p)        
        
        skill[idx] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def init_all_meta(self):
        skill = np.eye(self.max_skill_dim, dtype=np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_D_delta(self, skill, D_delta, step=None):
        
        if skill.shape[0] == self.batch_size:
            for idx in range(self.batch_size):
                self.delta_dict[np.argmax(skill[idx].cpu()).item()].append(D_delta[idx])
        else:  
            self.delta_dict[np.argmax(skill)].append(D_delta)

    def extend_skill(self):
        self.skill_dim += self.extend_num

    def update_D_delta_target(self, D_delta_all):
        self.D_delta_all = D_delta_all

    def plot(self, env, trajectory, save_dir, step, use_wandb, guide_plot = False, terminal_skill = False):

        if guide_plot:
            guide_idx = self.guide_skill.argmax()
            self.plot_guide(env = env,
                            trajectory = trajectory, 
                            save_dir = save_dir,
                            step = step,
                            use_wandb = use_wandb,
                            guide_idx = guide_idx.item(),
                            )

        else:
            if self.guide_skill is None: 
                self.plot_trajectory(env = env,
                                    trajectory = trajectory, 
                                    save_dir = save_dir,
                                    step = step,
                                    use_wandb = use_wandb,
                                    delta_dict = self.delta_dict)
                                
            else:
                guide_idx = self.guide_skill.argmax()
                self.plot_trajectory(env = env, 
                                    trajectory = trajectory, 
                                    save_dir = save_dir,
                                    step = step,
                                    use_wandb = use_wandb,
                                    guide_idx = guide_idx.item(),
                                    delta_dict = self.delta_dict)

    def initialize_extend_skill(self, guide_skill, exist = False):
        
        actor_guide_column_idx = torch.cat((torch.zeros(self.obs_dim - self.max_skill_dim), guide_skill.cpu())).to(self.device)
        actor_guide_column = torch.matmul(self.actor.trunk[0].weight, actor_guide_column_idx)

        critic_guide_column_idx = torch.cat((actor_guide_column_idx, torch.zeros(self.action_dim).to(self.device)))
        critic_guide_column = torch.matmul(self.critic.trunk[0].weight, critic_guide_column_idx)
        
        if exist is False:

            for idx in range(self.extend_num):
                sk_idx = self.obs_dim - self.max_skill_dim + self.skill_dim + idx
                
                self.actor.trunk[0].weight[:,sk_idx].data *= torch.zeros(self.hidden_dim).cuda()
                self.actor.trunk[0].weight[:,sk_idx].data += actor_guide_column
                self.critic.trunk[0].weight[:,sk_idx].data *= torch.zeros(self.hidden_dim).cuda()
                self.critic.trunk[0].weight[:,sk_idx].data += critic_guide_column

        else:
            
            for idx in range(self.extend_num):
                
                extended_sk_idx = self.skill_dim - self.extend_num + idx
                if (sum(self.delta_dict[extended_sk_idx]) / len(self.delta_dict[extended_sk_idx])) != 0:
                    sk_idx = self.obs_dim - self.max_skill_dim + extended_sk_idx

                    self.actor.trunk[0].weight[:,sk_idx].data *= torch.zeros(self.hidden_dim).cuda()
                    self.actor.trunk[0].weight[:,sk_idx].data += actor_guide_column
                    self.critic.trunk[0].weight[:,sk_idx].data *= torch.zeros(self.hidden_dim).cuda()
                    self.critic.trunk[0].weight[:,sk_idx].data += critic_guide_column

    def update_diayn(self, skill, next_obs, step = None):
        metrics = dict()

        loss, df_accuracy = self.compute_discriminator_loss(next_obs, skill)

        self.discriminator_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
            metrics['diayn_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, next_obs, step=None):
        if self.maze_parent in ['ant', '2dgym']:
            next_obs = next_obs[:, :2]

        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.discriminator(next_obs)
        d_pred[:, self.skill_dim:] = float('-inf')  
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)
        return reward * self.diayn_scale

    def compute_discriminator_loss(self, next_state, skill):
        """
        DF Loss
        """
        if self.maze_parent in ['ant', '2dgym']:
            next_state = next_state[:, :2]
            
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.discriminator(next_state)
        d_pred[:, self.skill_dim:] = float('-inf')  
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.discriminator_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy

    def compute_delta(self, obs, skill, test = False):
        
        if obs.shape[0] != self.batch_size:
            obs = torch.tensor(obs).unsqueeze(0).to(self.device)
            skill = torch.tensor(skill).unsqueeze(0).to(self.device)

        with torch.no_grad():
            d_pred = self.discriminator(obs)
            d_pred[:, self.skill_dim:] = float('-inf')
        
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        D_acc_logit = torch.exp(d_pred_log_softmax)
        
        D_acc = D_acc_logit[torch.arange(skill.shape[0]), skill.argmax(dim=1)].unsqueeze(1)
        mask = torch.where(D_acc > self.threshold, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
        
        D_delta = (1 - D_acc) * mask

        D_delta_target = D_delta if self.D_delta_all is None else torch.matmul(skill, self.D_delta_all)
        D_delta_target_mask = torch.where(D_delta_target > 0.0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

        D_delta_target = (1 - D_acc) * mask * D_delta_target_mask
        
        return D_delta, D_delta_target

    def find_guide(self, env, step):
        
        meta_all = self.init_all_meta()
        meta, observations, trajectory = OrderedDict(), OrderedDict(), OrderedDict()

        diffuse_timestep = int(self.maximum_timestep / 5)
        how_many_samples = 5

        for sk_idx in range(self.skill_dim):
            meta['skill'] = meta_all['skill'][sk_idx]
            observations[f'{sk_idx}'], trajectory[sk_idx] = list(), list()
            for idx in range(self.num_eval_each_skill):
                ts = env.reset()
                while not ts.last():
                    prev_obs = ts.observation

                    with torch.no_grad():
                        action = self.act(ts.observation,
                                          meta,
                                          step,
                                          eval_mode=True)
                    ts = env.step(action)

                    if idx == 0:
                        trajectory[sk_idx].append([[prev_obs[0].item(), ts.observation[0].item()], 
                                            [prev_obs[1].item(), ts.observation[1].item()]])

                origin = ts.observation
                for _ in range(diffuse_timestep):
                    
                    if (ts.observation[0] > self.min_max[0]) and  (ts.observation[0] < self.min_max[1]) and (ts.observation[1] > self.min_max[2]) and (ts.observation[1] < self.min_max[3]):
                        observations[f'{sk_idx}'].append(ts.observation[0:2])
                    else:
                        observations[f'{sk_idx}'].append(origin[0:2]) 
                    random_action = np.random.uniform(low = -env.action_range(), 
                                                    high = env.action_range(), size=(self.action_dim,))
                    
                    ts = env.step(random_action)

        all_obs = [(x,y) for i in range(len(observations)) for x,y in observations[str(i)]]
        all_obs = np.array(all_obs)
        all_samples = np.split(all_obs, len(observations), axis=0)

        neigh = NearestNeighbors(n_neighbors = how_many_samples)
        neigh.fit(all_obs)

        mean_distance = list()
        for idx in range(all_obs.shape[0]):
            x = neigh.kneighbors(all_obs[idx].reshape(1, -1), return_distance=True)
            mean_distance.append(x[0].mean())

        top_10_idx = np.argsort(mean_distance)[-1:]
        
        counts = np.bincount((top_10_idx/(self.num_eval_each_skill * diffuse_timestep)).astype(np.int))
        guide_skill_idx = np.argmax(counts)
        
        guide_skill = meta_all['skill'][guide_skill_idx]
        guide_skill = torch.tensor(guide_skill).to(self.device)

        save_dir = self.save_dir + '/knn_{}.png'.format(step)

        env.plot_knn(top1 = guide_skill_idx,
                        observations = observations,
                        all_samples= all_samples,
                        save_dir = save_dir,
                        use_wandb = True,
                        skill_dim = self.skill_dim)
        
        return guide_skill, trajectory

    def check_extend(self, env, step):
        count = 0
        D_delta_all = torch.ones(self.max_skill_dim, 1).to(self.device)
        for sk_idx in range(self.max_skill_dim):
            if self.D_delta_all[sk_idx] == 0:
                D_delta_all[sk_idx] = 0.0
            else:
                if sk_idx < self.skill_dim:
                    if (sum(self.delta_dict[sk_idx]) / len(self.delta_dict[sk_idx])).item() != 0:
                        count += 1

                D_delta_all[sk_idx] = sum(self.delta_dict[sk_idx]) / len(self.delta_dict[sk_idx])

        # Select guide and extend skill
        if count <= self.delta_threshold:
            self.extended_step = step
            self.guide_skill, trajectory = self.find_guide(env, step)
            save_dir = self.save_dir + '/guide_{}.png'.format(step)
            self.plot(env = env, trajectory = trajectory, save_dir = save_dir, step = step, use_wandb = True, guide_plot=True)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.initialize_extend_skill(self.guide_skill)
            self.update_D_delta_target(D_delta_all)
            self.extend_skill()
        
        # Just select guide
        elif ((step - self.extended_step) > self.interval) and self.guide_skill is not None and count <= self.delta_threshold + self.extend_num:
            self.extended_step = step
            self.guide_skill, trajectory = self.find_guide(env, step)
            save_dir = self.save_dir + '/guide_{}.png'.format(step)
            self.plot(env = env, trajectory = trajectory, save_dir = save_dir, step = step, use_wandb = True, guide_plot=True)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.initialize_extend_skill(self.guide_skill)
            self.update_D_delta_target(D_delta_all)
            self.extend_skill()
            self.delta_threshold += self.extend_num

        return count

    def update_critic(self, obs, action, reward, discount, next_obs, guide_next_obs, guide_skill, D_delta, step):
        # obs: (s, z)
        # next_obs: (s', z)
        # guide_next_obs: (s', z*) 
        # guide_skill: (z*)
        metrics = dict()

        next_action, D_KL, log_prob = self.sample_action_and_KL(next_obs, guide_next_obs, guide_skill)

        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        if guide_skill is not None:
            target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach() * log_prob) - (self.beta.detach() * D_delta * D_KL)
        else:
            target_V = torch.min(target_Q1, target_Q2) - (self.alpha.detach() * log_prob)
        target_Q = reward + (discount * target_V)
        target_Q = target_Q.detach()

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                        F.mse_loss(current_Q2, target_Q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        mask = torch.where(D_delta > 0,torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = current_Q1.mean().item()
            metrics['critic_q2'] = current_Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['log_prob'] = log_prob.mean().item()
            metrics['Next_D_KL'] = (D_KL * mask).mean().item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad()
            self.encoder_opt.step()

        return metrics

    # update actor for DISCO
    def update_actor(self, obs, guide_obs, guide_skill, D_delta, step):
        # obs: (s, z)
        # guide_obs: (s, z*)
        # guide_skill: (z*)
        metrics = dict()

        action, D_KL, log_prob = self.sample_action_and_KL(obs, guide_obs, guide_skill)
        
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # optimize alpha
        if self.use_auto_alpha:
            self.log_alpha_opt.zero_grad()
            alpha_loss.backward()
            self.log_alpha_opt.step()
            
        mask = torch.where(D_delta > 0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['alpha_loss'] = alpha_loss.item()
            metrics['alpha_value'] = self.alpha.item()
            metrics['D_KL'] = (D_KL * mask).mean().item()
            
        return metrics

    def update(self, replay_iter, env, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        if (step % self.check_delta_every_steps == 0) and (step != 0):
            self.count = self.check_extend(env, step)

        batch = next(replay_iter)

        terminal, obs, action, extr_reward, discount, next_obs, xy_p, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        terminal = self.aug_and_encode(terminal)
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        D_delta, D_delta_target = self.compute_delta(terminal, skill, test=True)
        
        if self.reward_free:

            metrics.update(self.update_diayn(skill, next_obs, step))
            
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()
            metrics['D_delta'] = D_delta.mean().item()
            metrics['D_delta_target'] = D_delta_target.mean().item()
            metrics['clamp_kl'] = self.clamp_kl
            metrics['count'] = self.count

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        if self.guide_skill is None:
            guide_skill = self.guide_skill
            guide_obs, guide_next_obs = torch.zeros_like(obs), torch.zeros_like(obs)
        else:
            guide_skill = self.guide_skill.unsqueeze(0).repeat(skill.shape[0],1)
            guide_obs = torch.cat([obs, guide_skill], dim=1)
            guide_next_obs = torch.cat([next_obs, guide_skill], dim=1)     

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount, next_obs.detach(), 
                               guide_next_obs, guide_skill, D_delta_target, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), guide_obs,
                                         guide_skill, D_delta_target, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def get_dir(self, file_name):
        save_dir = self.work_dir / 'eval_video'
        dir_name = file_name.split('/')[0]
        if not os.path.exists(save_dir / dir_name):
            os.makedirs(save_dir / dir_name)
        path = save_dir / file_name

        return path