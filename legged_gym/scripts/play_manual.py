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
    lin_x = 1.5  # m/s
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
    train_cfg.runner.resume = True  # load policy weights via args/load_run
    # Prevent CLI default ("debug") from overwriting runner class in cfg.
    args.runner_class_name = None

    # Prepare environment and policy.
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Disable internal random command resampling to keep full manual control.
    def _noop_resample(env_ids):
        return
    env._resample_commands = _noop_resample  # type: ignore

    obs_dict = env.reset()

    # Логгер для визуализации графиков.
    from legged_gym.utils import Logger
    logger = Logger(env.dt)
    selected_joint_names = [
        "FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint",
        "FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint",
    ]
    dof_names = list(getattr(env, 'dof_names', []))
    selected_dof_indices = []
    missing = []
    for n in selected_joint_names:
        if n in dof_names:
            selected_dof_indices.append(dof_names.index(n))
        else:
            missing.append(n)
    if len(missing) > 0:
        print("Logger selection warning, some requested joints not found in env.dof_names:", missing)
    logger.log_state('dof_names_sel', selected_joint_names)
    robot_index = 0
    stop_state_log = 1000
    stop_rew_log = env.max_episode_length + 1

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
    for i in range(10 * int(env.max_episode_length)):
        ep_step = int(env.episode_length_buf[0].item())
        lin_x, lin_y, yaw_rate = control_cmd(ep_step, env.max_episode_length)
        env.commands[:, 0] = lin_x
        env.commands[:, 1] = lin_y
        env.commands[:, 2] = yaw_rate
        # env.action_residual[:, :] = torch.tensor([...], device=env.device)
        env.compute_observations()
        obs_dict = env.get_observations()

        actions = policy(obs_dict)
        obs_dict, rews, dones, infos, _, _ = env.step(actions.detach())

        # Логгирование данных для графиков
        try:
            full_tgt = (actions[robot_index].detach().cpu().numpy() * env.cfg.control.action_scale)
        except Exception:
            full_tgt = (actions[robot_index].cpu().numpy() * env.cfg.control.action_scale)
        try:
            full_pos = env.dof_pos[robot_index].detach().cpu().numpy() if hasattr(env.dof_pos[robot_index], 'detach') else env.dof_pos[robot_index].cpu().numpy()
        except Exception:
            full_pos = np.array(env.dof_pos[robot_index])
        try:
            full_vel = env.dof_vel[robot_index].detach().cpu().numpy() if hasattr(env.dof_vel[robot_index], 'detach') else env.dof_vel[robot_index].cpu().numpy()
        except Exception:
            full_vel = np.array(env.dof_vel[robot_index])
        try:
            full_torque = env.torques[robot_index].detach().cpu().numpy() if hasattr(env.torques[robot_index], 'detach') else env.torques[robot_index].cpu().numpy()
        except Exception:
            full_torque = np.array(env.torques[robot_index])

        sel_tgt = np.array([full_tgt[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])
        sel_pos = np.array([full_pos[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])
        sel_vel = np.array([full_vel[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])
        sel_torque = np.array([full_torque[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])

        logger.log_states(
            {
                'dof_pos_target': sel_tgt,
                'dof_pos': sel_pos,
                'dof_vel': sel_vel,
                'dof_torque': sel_torque,
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            }
        )
        if i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos.get("episode"):
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    args = get_args()
    play_manual(args)
