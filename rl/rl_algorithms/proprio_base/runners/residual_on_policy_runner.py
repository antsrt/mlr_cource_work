# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import time
import os
from collections import deque
import statistics
import json

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import distutils.version

from torch.utils.tensorboard import SummaryWriter
import torch
import csv
import json

from ..algorithms.residual_ppo import ResidualPPO
from ..modules.actor_critic_residual import ActorCriticResidual
from rl.env import VecEnv


class ResidualOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic = ActorCriticResidual(
            num_actor_obs=self.env.num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        self.alg: ResidualPPO = ResidualPPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        hist_shape = [self.env.num_obs * self.env.include_history_steps] if self.env.include_history_steps is not None else [self.env.num_obs]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            hist_shape,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        # Запомнить базовые коэффициенты для восстановления после warmup.
        self.base_kl_default = getattr(self.alg, "base_kl_coef", None)
        self.residual_l2_default = getattr(self.alg, "residual_l2_coef", None)
        self.std_default = self.alg.actor_critic.std.detach().clone()

        self.env.reset()
        # Сохранить параметры запуска рядом с логами.
        if self.log_dir is not None:
            def _to_serializable(obj):
                if hasattr(obj, "tolist"):
                    try:
                        return obj.tolist()
                    except Exception:
                        return float(obj)
                if isinstance(obj, (int, float, str, bool)):
                    return obj
                return str(obj)

            params = {
                "train_cfg": self.cfg,
                "algorithm_cfg": self.alg_cfg,
                "policy_cfg": self.policy_cfg,
                "command_ranges": {k: _to_serializable(v) for k, v in self.env.command_ranges.items()},
            }
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "params.json"), "w") as f:
                json.dump(params, f, indent=2)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs_dict = self.env.get_observations()
        obs = obs_dict["obs"].to(self.device)
        critic_obs = obs_dict["privileged_obs"].to(self.device) if obs_dict["privileged_obs"] is not None else obs
        obs_device = {"obs": obs, "privileged_obs": critic_obs, "proprio_hist": obs_dict["proprio_hist"].to(self.device)}

        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # Warmup: выше шум и слабее KL/L2 на старте.
            warm_iters = self.cfg.get("warmup_iters", 0)
            if it < warm_iters:
                warm_std = self.cfg.get("warmup_std", None)
                if warm_std is not None:
                    with torch.no_grad():
                        self.alg.actor_critic.std.data.fill_(warm_std)
                if self.base_kl_default is not None:
                    self.alg.base_kl_coef = self.cfg.get("warmup_kl_coef", self.base_kl_default)
                if self.residual_l2_default is not None:
                    self.alg.residual_l2_coef = self.cfg.get("warmup_l2_coef", self.residual_l2_default)
            else:
                # Вернуть базовые значения после warmup.
                with torch.no_grad():
                    self.alg.actor_critic.std.data.copy_(self.std_default)
                if self.base_kl_default is not None:
                    self.alg.base_kl_coef = self.base_kl_default
                if self.residual_l2_default is not None:
                    self.alg.residual_l2_coef = self.residual_l2_default

            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_device)
                    obs_dict, rewards, dones, infos, _, _ = self.env.step(actions)
                    obs = obs_dict["obs"].to(self.device)
                    critic_obs = obs_dict["privileged_obs"].to(self.device) if obs_dict["privileged_obs"] is not None else obs
                    obs_device = {"obs": obs, "privileged_obs": critic_obs, "proprio_hist": obs_dict["proprio_hist"].to(self.device)}
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs_device)

            mean_value_loss, mean_surrogate_loss, mean_kl = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals(), mean_kl)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, mean_kl, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        # Сохранение метрик в файл рядом с output_name (log_dir).
        if self.log_dir is not None and locs["it"] % 10 == 0:
            csv_path = os.path.join(self.log_dir, "metrics.csv")
            file_exists = os.path.exists(csv_path)
            reward_mean = statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) > 0 else float("nan")
            ep_len_mean = statistics.mean(locs["lenbuffer"]) if len(locs["lenbuffer"]) > 0 else float("nan")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "iteration",
                            "timesteps",
                            "fps",
                            "collection_time",
                            "learn_time",
                            "value_loss",
                            "surrogate_loss",
                            "kl_to_base",
                            "mean_noise_std",
                            "mean_reward",
                            "mean_episode_length",
                        ]
                    )
                writer.writerow(
                    [
                        locs["it"],
                        self.tot_timesteps,
                        int(self.num_steps_per_env * self.env.num_envs / iteration_time),
                        locs["collection_time"],
                        locs["learn_time"],
                        locs["mean_value_loss"],
                        locs["mean_surrogate_loss"],
                        mean_kl,
                        self.alg.actor_critic.std.mean().item(),
                        reward_mean,
                        ep_len_mean,
                    ]
                )

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/KL_base", mean_kl, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if locs["it"] % 10 == 0:
            if len(locs["rewbuffer"]) > 0:
                log_string = (
                    f"""{'#' * width}\n"""
                    f"""{header.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'KL to base:':>{pad}} {mean_kl:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                    f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                    f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                )
            else:
                log_string = (
                    f"""{'#' * width}\n"""
                    f"""{header.center(width, ' ')}\n\n"""
                    f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                    f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                    f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                    f"""{'KL to base:':>{pad}} {mean_kl:.4f}\n"""
                    f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                )

            log_string += (
                f"""{'-' * width}\n"""
                f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
            )
            print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        # Совместимость со старыми чекпойнтами, где std сохранялась как action_std.
        state_dict = loaded_dict["model_state_dict"]
        if "action_std" in state_dict and "std" not in state_dict:
            state_dict["std"] = state_dict.pop("action_std")
        self.alg.actor_critic.load_state_dict(state_dict, strict=False)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
