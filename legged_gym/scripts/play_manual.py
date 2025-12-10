# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Lightweight inference script that mirrors `play.py`, but replaces joystick
# control with a user-defined command generator. Edit `control_cmd` to shape
# the velocity commands you want to feed into the policy at each step.

import os
from typing import Tuple

import isaacgym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def control_cmd(step: int, max_steps: int) -> Tuple[float, float, float]:
    """
    User-editable command generator.
    Args:
        step: current simulation step (0-based).
        max_steps: total planned steps for this rollout.
    Returns:
        (lin_vel_x, lin_vel_y, ang_vel_yaw)
    """
    # Example: constant forward command with zero yaw.
    lin_x = 2.0  # m/s
    lin_y = 0.0  # m/s
    yaw_rate = 0.0  # rad/s
    return lin_x, lin_y, yaw_rate

# lin vel x: -0.211 .. 1.451 м/с фактически максимум 1.1 м/с
# lin vel y: -0.421 .. 0.806 м/с фактически максимум 2.0 м/с точно, больше не тестил
# ang vel z: -1.818 .. 1.197 рад/с при 0.6 идет прямо

def play_manual(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override for deterministic single-env inference.
    env_cfg.env.num_envs = 1
    env_cfg.env.get_commands_from_joystick = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.resampling_time = 1e9  # effectively disable random resampling

    # AMP data preload kept minimal for inference.
    train_cfg.runner.amp_num_preload_transitions = 1
    train_cfg.runner.resume = True  # always load policy weights
    # Prevent CLI default ("debug") from overwriting runner class in cfg.
    args.runner_class_name = None

    # Prepare environment and policy.
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Disable internal random command resampling to keep full manual control.
    def _noop_resample(env_ids):
        return
    env._resample_commands = _noop_resample  # type: ignore

    obs_dict = env.reset()

    # Лог для сравнения команд и фактических скоростей.
    log_lin_vel = []
    log_ang_vel = []
    log_cmds = []

    # Manual residual hook: set this to shape (num_envs, num_actions).
    env.action_residual[:] = 0.0

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    max_steps = int(1 * env.max_episode_length)
    # Align initial observations with our commands before the first policy call.
    lin_x, lin_y, yaw_rate = control_cmd(0, env.max_episode_length)
    env.commands[:, 0] = lin_x
    env.commands[:, 1] = lin_y
    env.commands[:, 2] = yaw_rate
    env.compute_observations()
    obs_dict = env.get_observations()
    for step in range(max_steps):
        # Per-episode step to avoid slow global ramps.
        ep_step = int(env.episode_length_buf[0].item())
        lin_x, lin_y, yaw_rate = control_cmd(ep_step, env.max_episode_length)
        env.commands[:, 0] = lin_x
        env.commands[:, 1] = lin_y
        env.commands[:, 2] = yaw_rate
        # Example manual residual: keep zero unless you want to adjust actions.
        # env.action_residual[:, :] = torch.tensor([...], device=env.device)
        env.compute_observations()
        obs_dict = env.get_observations()

        actions = policy(obs_dict)
        obs_dict, rews, dones, infos, _, _ = env.step(actions.detach())

        # Сбор логов: фактические и целевые скорости.
        log_lin_vel.append(env.base_lin_vel[0, :2].cpu().numpy())
        log_ang_vel.append(env.base_ang_vel[0, 2].cpu().numpy())
        log_cmds.append(env.commands[0, :3].cpu().numpy())

        # Optional: print rewards when an episode ends.
        if infos.get("episode") and torch.sum(env.reset_buf).item() > 0:
            print(f"Step {step}: episode rewards {infos['episode']}")

    # Построить и сохранить графики ошибок после прогона.
    if len(log_cmds) > 0:
        log_lin_vel = np.stack(log_lin_vel)
        log_ang_vel = np.stack(log_ang_vel)
        log_cmds = np.stack(log_cmds)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        t = np.arange(len(log_cmds))
        axes[0].plot(t, log_cmds[:, 0], label="cmd lin x", linestyle="--")
        axes[0].plot(t, log_cmds[:, 1], label="cmd lin y", linestyle="--")
        axes[0].plot(t, log_lin_vel[:, 0], label="meas lin x")
        axes[0].plot(t, log_lin_vel[:, 1], label="meas lin y")
        axes[0].set_ylabel("Linear vel (m/s)")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(t, log_cmds[:, 2], label="cmd yaw", linestyle="--")
        axes[1].plot(t, log_ang_vel, label="meas yaw")
        axes[1].set_ylabel("Yaw vel (rad/s)")
        axes[1].set_xlabel("Step")
        axes[1].legend()
        axes[1].grid(True)

        out_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "vel_tracking.png")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved velocity tracking plot to {out_path}")


if __name__ == "__main__":
    args = get_args()
    play_manual(args)
