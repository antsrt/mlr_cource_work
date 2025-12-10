# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Residual policy: frozen base actor + trainable residual head.

import torch
import torch.nn as nn
from torch.distributions import Normal
from termcolor import cprint

from legged_gym import LEGGED_GYM_ROOT_DIR
from .actor_critic import get_activation


class ActorCriticResidual(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0,
        residual_hidden_dims=[256, 128],
        residual_scale=2.0,
        base_checkpoint=None,
        **kwargs,
    ):
        super().__init__()
        # Базовая политика была обучена с сигмоидой внутри ActorCritic, поэтому
        # фиксируем ту же активацию, чтобы загруженные веса совпадали по поведению.
        act = get_activation('sigmoid')

        self.num_actor_input = num_actor_obs
        self.num_critic_input = num_critic_obs
        self.residual_scale = residual_scale

        # Base actor (frozen)
        base_layers = [nn.Linear(self.num_actor_input, actor_hidden_dims[0]), act]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                base_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                base_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                base_layers.append(act)
        self.base_actor = nn.Sequential(*base_layers)
        for p in self.base_actor.parameters():
            p.requires_grad = False

        # Residual head (trainable)
        res_layers = []
        in_dim = self.num_actor_input
        for h in residual_hidden_dims:
            res_layers.append(nn.Linear(in_dim, h))
            res_layers.append(nn.ReLU())
            in_dim = h
        res_layers.append(nn.Linear(in_dim, num_actions))
        self.residual_head = nn.Sequential(*res_layers)

        # Critic
        critic_layers = [nn.Linear(self.num_critic_input, critic_hidden_dims[0]), act]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(act)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise (shared for total)
        std = init_noise_std * torch.ones(num_actions)
        self.std = nn.Parameter(std)
        self.distribution = None

        # Load base weights
        self.base_action_std = torch.ones(num_actions) * init_noise_std
        if base_checkpoint is not None:
            path = base_checkpoint.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            ckpt = torch.load(path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            remap = {k.replace('actor.', '', 1): v for k, v in sd.items() if k.startswith('actor.')}
            missing, unexpected = self.base_actor.load_state_dict(remap, strict=False)
            if unexpected:
                cprint(f"Unexpected keys in base load: {unexpected}", 'yellow')
            if missing:
                cprint(f"Missing keys in base load: {missing}", 'yellow')
            base_std = sd.get('std', None)
            if base_std is not None:
                self.base_action_std = base_std.clone().detach()
        self.register_buffer('base_action_std_buf', self.base_action_std.clone())
        self.base_action_mean = None
        self.residual_action_mean = None

    def reset(self, dones=None):
        pass

    def act(self, obs_dict, **kwargs):
        obs = obs_dict['obs']
        base_mu = self.base_actor(obs)
        # Residual без tanh, чтобы диапазон совпадал с базой; масштабируем коэффициентом.
        res_mu = self.residual_head(obs) * self.residual_scale
        self.base_action_mean = base_mu
        self.residual_action_mean = res_mu
        # Noise only on residual component; base stays deterministic.
        self.action_std = self.std.to(res_mu.device)
        self.distribution = Normal(res_mu, res_mu * 0.0 + self.action_std)
        total_action = base_mu + self.distribution.sample()
        self.action_mean = base_mu + res_mu
        return total_action

    def get_actions_log_prob(self, actions):
        # Log-prob only for residual part; base is deterministic.
        residual_actions = actions - self.base_action_mean
        return self.distribution.log_prob(residual_actions).sum(dim=-1)

    def act_inference(self, obs_dict):
        # Deterministic inference: return mean without sampling noise.
        obs = obs_dict['obs']
        base_mu = self.base_actor(obs)
        res_mu = self.residual_head(obs) * self.residual_scale
        total_mu = base_mu + res_mu
        # Track for logging/compat.
        self.base_action_mean = base_mu
        self.action_mean = total_mu
        self.action_std = torch.zeros_like(self.std, device=total_mu.device)
        return total_mu

    def evaluate(self, obs_dict, **kwargs):
        obs_privileged = obs_dict['privileged_obs']
        return self.critic(obs_privileged)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
