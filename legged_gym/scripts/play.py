# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.get_commands_from_joystick = True
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False

    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _ = env.reset()
    obs_dict = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    # Select only the 8 joints requested (2 per leg: calf and the preceding thigh)
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
    # store the selected names for the logger (so plots can use readable labels)
    logger.log_state('dof_names_sel', selected_joint_names)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 500 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # --- speed ramp settings: change linear X command starting at t = ramp_start_s ---
    ramp_start_s = 5.0
    ramp_duration_s = 3.0
    target_lin_vel_x = 1.0
    # disable joystick input (if present) and automatic resampling so our manual commands persist
    try:
        if hasattr(env, '_get_commands_from_joystick'):
            env._get_commands_from_joystick = False
    except Exception:
        pass
    try:
        # set a very large resampling time to avoid overwriting commands
        if hasattr(env.cfg.commands, 'resampling_time'):
            env.cfg.commands.resampling_time = 1e9
    except Exception:
        pass
    # record initial command (use robot_index as reference)
    try:
        initial_lin_vel = float(env.commands[robot_index, 0].cpu().numpy())
    except Exception:
        try:
            initial_lin_vel = float(env.commands[robot_index, 0].item())
        except Exception:
            initial_lin_vel = 0.0
    ramp_done = False

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs_dict)
        obs_dict, rews, dones, infos, _, _ = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # handle speed ramping (smooth change of desired linear velocity in X)
        if not ramp_done:
            current_time = i * env.dt
            if current_time >= ramp_start_s:
                frac = min(1.0, (current_time - ramp_start_s) / max(1e-6, ramp_duration_s))
                new_val = initial_lin_vel + (target_lin_vel_x - initial_lin_vel) * frac
                # assign to all envs (safe cast to tensor on env device)
                try:
                    env.commands[:, 0] = torch.full((env.num_envs,), float(new_val), device=env.device)
                except Exception:
                    try:
                        env.commands[:, 0] = float(new_val)
                    except Exception:
                        pass
                if frac >= 1.0:
                    ramp_done = True
                    print(f"Speed ramp complete: lin_vel_x -> {new_val} at t={current_time:.2f}s (i={i})")

        if i < stop_state_log:
            # compute target and measured vectors only for selected DOFs
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
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
