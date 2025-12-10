# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# PPO variant for residual policy with KL to frozen base on high-speed commands.

import torch
import torch.nn as nn
import torch.optim as optim

from . import ppo
from ..storage.rollout_storage_residual import RolloutStorageResidual


class ResidualPPO(ppo.PPO):
    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        base_kl_coef=0.1,
        speed_threshold=1.0,
        residual_l2_coef=0.0,
        device="cpu",
    ):
        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
        )
        self.base_kl_coef = base_kl_coef
        self.speed_threshold = speed_threshold
        self.residual_l2_coef = residual_l2_coef

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, hist_info_shape):
        self.storage = RolloutStorageResidual(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, hist_info_shape, self.device
        )

    def act(self, obs_dict):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs_dict).detach()
        self.transition.values = self.actor_critic.evaluate(obs_dict).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # base stats
        self.transition.base_action_mean = getattr(self.actor_critic, "base_action_mean", None)
        self.transition.base_action_sigma = getattr(self.actor_critic, "base_action_std_buf", None)
        self.transition.observations = obs_dict["obs"]
        self.transition.critic_observations = obs_dict["privileged_obs"]
        self.transition.proprio_hist = obs_dict["proprio_hist"]
        return self.transition.actions

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            base_mu_batch,
            base_sigma_batch,
            hid_states_batch,
            masks_batch,
            proprio_hist_batch,
        ) in generator:

            obs_dict_batch = {
                "obs": obs_batch,
                "privileged_obs": critic_obs_batch,
                "proprio_hist": proprio_hist_batch,
            }

            self.actor_critic.act(obs_dict_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_dict_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # KL to base on high-speed commands (mask)
            kl_mask = torch.ones_like(mu_batch[:, 0], dtype=torch.bool)
            # commands are at indices 9:12 in critic_obs (scaled)
            if critic_obs_batch is not None and critic_obs_batch.shape[1] >= 12:
                cmds_scaled = critic_obs_batch[:, 9:12]
                cmd_scale = torch.tensor([2.0, 2.0, 0.25], device=cmds_scaled.device)
                cmd_unscaled = cmds_scaled / cmd_scale
                speed = torch.linalg.norm(cmd_unscaled[:, :2], dim=-1)
                kl_mask = speed > self.speed_threshold
            safe_sigma = torch.clamp(sigma_batch, min=1e-5)
            safe_base_sigma = torch.clamp(base_sigma_batch, min=1e-5)
            kl_terms = torch.log(safe_sigma / safe_base_sigma) + (
                torch.square(safe_base_sigma) + torch.square(base_mu_batch - mu_batch)
            ) / (2.0 * torch.square(safe_sigma)) - 0.5
            kl_all = torch.sum(kl_terms, dim=-1)
            if kl_mask.any():
                kl_to_base = kl_all[kl_mask].mean()
            else:
                kl_to_base = torch.tensor(0.0, device=mu_batch.device)

            # L2 penalty on residual to keep it small
            residual = mu_batch - base_mu_batch
            residual_penalty = torch.mean(torch.square(residual))

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.base_kl_coef * kl_to_base + self.residual_l2_coef * residual_penalty

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_to_base.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_kl
