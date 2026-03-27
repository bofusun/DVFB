# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb  # pylint: disable=unused-import
import math
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
import utils


class OnlineCov(nn.Module):
    def __init__(self, mom: float, dim: int) -> None:
        super().__init__()
        self.mom = mom  # momentum
        self.count = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)
        self.cov: tp.Any = torch.nn.Parameter(torch.zeros((dim, dim), dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.count += 1  # type: ignore
            self.cov.data *= self.mom
            self.cov.data += (1 - self.mom) * torch.matmul(x.T, x) / x.shape[0]
        count = self.count.item()
        cov = self.cov / (1 - self.mom**count)
        return cov


class _L2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        y = math.sqrt(self.dim) * F.normalize(x, dim=1)
        return y


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension"""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Provides a sequence of linear layers and non-linearities
    providing a sequence of dimension for the neurons, or name of
    the non-linearities
    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)


class Actor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", self.action_dim)
        self.apply(utils.weight_init)
        # initialize the last layer by zero
        # self.policy[-1].weight.data.fill_(0.0)

    def forward(self, obs, z, std):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, log_std_bounds,
                 preprocess=False) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.log_std_bounds = log_std_bounds
        self.preprocess = preprocess
        feature_dim = obs_dim + z_dim

        self.policy = mlp(feature_dim, hidden_dim, "ntanh", hidden_dim, "relu", 2 * action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z):
        assert z.shape[-1] == self.z_dim
        h = torch.cat([obs, z], dim=-1)
        mu, log_std = self.policy(h).chunk(2, dim=-1)
        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = utils.SquashedNormal(mu, std)
        return dist


class ForwardMap(nn.Module):
    """ forward representation class"""

    def __init__(self, obs_dim, z_dim, action_dim, feature_dim, hidden_dim,
                 preprocess=False, add_trunk=True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(self.obs_dim + self.action_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(self.obs_dim + self.z_dim + self.action_dim, hidden_dim, "ntanh",
                             hidden_dim, "irelu",
                             hidden_dim, "irelu")
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)

        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2

    
class IdentityMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.B = nn.Identity()

    def forward(self, obs):
        return self.B(obs)


class BackwardMap(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

class BackwardMap_cic(nn.Module):
    """ backward representation class"""

    def __init__(self, state_net, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.state_net = state_net
        self.B = mlp(self.z_dim, hidden_dim, "relu", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True
        with torch.no_grad():
            obs = self.state_net(obs)
        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

class BackwardMap_cic1(nn.Module):
    """ backward representation class"""

    def __init__(self, state_net, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.state_net = state_net
        self.B = mlp(self.z_dim, hidden_dim, "relu", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True
        with torch.no_grad():
            obs = self.state_net(obs)
        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B
    
class BackwardMap_dist(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", hidden_dim, "relu", 2*self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        B_mu, B_log_std = self.B(obs).chunk(2, dim=-1)
        if self.norm_z:
            B_mu = math.sqrt(self.z_dim) * F.normalize(B_mu, dim=1)
        B_log_std = torch.clamp(B_log_std, min=-20, max=2)
        B_std = torch.exp(B_log_std)
        dist = torch.distributions.Normal(B_mu, B_std)
        return dist
    
class BackwardMap_trunk(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, feature_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.trunk =  mlp(self.obs_dim, feature_dim, "ntanh")
        self.B = mlp(feature_dim, "relu",  hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        obs = self.trunk(obs)
        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

class BackwardMap_trans_trunk(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, feature_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.trunk =  mlp(self.obs_dim, feature_dim, "ntanh")
        self.B = mlp(2*feature_dim, "relu",  hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True
        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B
    
# 只归一化的版本
class BackwardMap1(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        B = self.B(obs)
        if self.norm_z:
            B = F.normalize(B, dim=1)
        return B

# 输入s和s'的版本
class BackwardMap2(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=-1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=-1)
        return B

# 输入s和s'的版本
class BackwardMap2_trunk(nn.Module):
    """ backward representation class"""

    def __init__(self, state_net, pred_net, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.state_net = state_net
        self.pred_net = pred_net
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.z_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        with torch.no_grad():
            state = self.state_net(obs)
            next_state = self.state_net(next_obs)
            inpt = self.pred_net(torch.cat([state,next_state],1))
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=-1)
        return B
    
# 输入s和s'的版本
class BackwardMap2_trunk1(nn.Module):
    """ backward representation class"""

    def __init__(self, state_net, pred_net, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.state_net = state_net
        self.pred_net = pred_net
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.z_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        state = self.state_net(obs)
        next_state = self.state_net(next_obs)
        inpt = self.pred_net(torch.cat([state,next_state],1))
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=-1)
        return B
    
# 输入s和s'的版本,加mlp
class BackwardMap3(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, feature_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.trunk= mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        self.B = mlp(2*feature_dim, hidden_dim, "relu", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本
class BackwardMap4(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = F.normalize(B, dim=1)
        return B

# 输入s和s'的版本
class BackwardMap5(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本,加mlp
class BackwardMap6(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, feature_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.trunk= mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        self.B = mlp(2*feature_dim, hidden_dim, "relu", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.trunk(obs)
        next_obs = self.trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本,加mlp
class BackwardMap7(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, feature_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.obs_trunk= mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        self.next_obs_trunk= mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        self.B = mlp(2*feature_dim, hidden_dim, "irelu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.obs_trunk(obs)
        next_obs = self.next_obs_trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本,加mlp
class BackwardMap8(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, feature_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.obs_trunk= mlp(self.obs_dim, feature_dim, "ntanh")
        self.next_obs_trunk= mlp(self.obs_dim, feature_dim, "ntanh")
        self.B = mlp(2*feature_dim, hidden_dim, "irelu", hidden_dim, "irelu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.obs_trunk(obs)
        next_obs = self.next_obs_trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本,加mlp
class BackwardMap9(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, feature_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.obs_trunk= mlp(self.obs_dim, hidden_dim, "irelu", feature_dim)
        self.next_obs_trunk= mlp(self.obs_dim, hidden_dim, "irelu", feature_dim)
        self.B = mlp(2*feature_dim, hidden_dim, "irelu", hidden_dim, "irelu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.obs_trunk(obs)
        next_obs = self.next_obs_trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本,加mlp
class BackwardMap10(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, feature_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.obs_trunk= mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        self.next_obs_trunk= mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
        self.B = mlp(2*feature_dim, hidden_dim, "irelu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        obs = self.obs_trunk(obs)
        next_obs = self.next_obs_trunk(next_obs)
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = F.normalize(B, dim=1)
        return B

# 输入s和s'的版本
class BackwardMap11(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = F.normalize(B, dim=1)
        return B

# 输入s和s'的版本
class BackwardMap12(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B

# 输入s和s'的版本
class BackwardMap13(nn.Module):
    """ backward representation class"""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(2*self.obs_dim, hidden_dim, "ntanh", hidden_dim, "tanh", hidden_dim, "tanh", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, next_obs):
        if not hasattr(self, "norm_z"):  # backward compatiblity
            self.norm_z = True

        inpt = torch.cat([obs, next_obs], dim=1)
        B = self.B(inpt)
        if self.norm_z:
            B = F.normalize(B, dim=1)
        return B
    
class MultinputNet(nn.Module):
    """Network with multiple inputs"""

    def __init__(self, input_dims: tp.Sequence[int], sequence_dims: tp.Sequence[int]) -> None:
        super().__init__()
        input_dims = list(input_dims)
        sequence_dims = list(sequence_dims)
        dim0 = sequence_dims[0]
        self.innets = nn.ModuleList([mlp(indim, dim0, "relu", dim0, "layernorm") for indim in input_dims])  # type: ignore
        sequence: tp.List[tp.Union[str, int]] = [dim0]
        for dim in sequence_dims[1:]:
            sequence.extend(["relu", dim])
        self.outnet = mlp(*sequence)  # type: ignore

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        assert len(tensors) == len(self.innets)
        out = sum(net(x) for net, x in zip(self.innets, tensors)) / len(self.innets)
        return self.outnet(out)  # type : ignore
