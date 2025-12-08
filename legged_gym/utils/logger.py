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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import os
import time

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        # Create simplified plots:
        # 1) Linear velocity plots: command_x vs base_vel_x and command_y vs base_vel_y
        # 2) Angular yaw velocity plot: command_yaw vs base_vel_yaw
        log = self.state_log

        # determine maximum logged length
        max_len = 0
        for v in log.values():
            if hasattr(v, '__len__'):
                max_len = max(max_len, len(v))
        if max_len == 0:
            return

        time = np.linspace(0, max_len * self.dt, max_len)

        def series(key):
            arr = log.get(key, [])
            if len(arr) == 0:
                return np.full(max_len, np.nan)
            a = np.array(arr)
            # if values are vectors, try to reduce to first component
            if a.ndim > 1:
                try:
                    a = a.reshape(a.shape[0], -1)
                    # if more than one column, take first
                    a = a[:, 0]
                except Exception:
                    a = a.flatten()[:max_len]
            if a.shape[0] < max_len:
                pad = np.full(max_len - a.shape[0], np.nan)
                a = np.concatenate([a, pad])
            return a

        def series_matrix(key):
            arr = log.get(key, [])
            if len(arr) == 0:
                return None
            # determine number of columns from first entry
            first = np.array(arr[0])
            if first.ndim == 0:
                return None
            ncols = first.reshape(-1).shape[0]
            mat = np.full((max_len, ncols), np.nan)
            for i, v in enumerate(arr):
                a = np.array(v).reshape(-1)
                cols = min(ncols, a.shape[0])
                mat[i, :cols] = a[:cols]
            return mat

        cmd_x = series('command_x')
        cmd_y = series('command_y')
        base_x = series('base_vel_x')
        base_y = series('base_vel_y')
        cmd_yaw = series('command_yaw')
        base_yaw = series('base_vel_yaw')

        # Single figure with 3 subplots: X, Y, yaw (joint plots are in a separate figure)
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # X component
        axs[0].plot(time, cmd_x, linestyle='--', label='command_x')
        axs[0].plot(time, base_x, label='base_vel_x')
        axs[0].set(xlabel='time [s]', ylabel='vel x [m/s]', title='Command vs Base Velocity (X)')
        axs[0].legend()

        # Y component
        axs[1].plot(time, cmd_y, linestyle='--', label='command_y')
        axs[1].plot(time, base_y, label='base_vel_y')
        axs[1].set(xlabel='time [s]', ylabel='vel y [m/s]', title='Command vs Base Velocity (Y)')
        axs[1].legend()

        # Yaw
        axs[2].plot(time, cmd_yaw, linestyle='--', label='command_yaw')
        axs[2].plot(time, base_yaw, label='base_vel_yaw')
        axs[2].set(xlabel='time [s]', ylabel='yaw vel [rad/s]', title='Command Yaw vs Base Yaw')
        axs[2].legend()

        # Joint positions: measured vs target — all joints together in one subplot
        # try to get human-readable joint names if the logger received them
        joint_names = None
        names_entry = log.get('dof_names_sel', None)
        if names_entry and len(names_entry) > 0:
            first = names_entry[0]
            try:
                if isinstance(first, (list, tuple, np.ndarray)):
                    joint_names = list(first)
                elif isinstance(first, str):
                    # single string (unlikely) -> treat as single name
                    joint_names = [first]
            except Exception:
                joint_names = None

        dof_target_mat = series_matrix('dof_pos_target')
        dof_pos_mat = series_matrix('dof_pos')

        # Create a separate joint-figure with one subplot per leg (FR, FL, RR, RL).
        fig_j = None
        if dof_target_mat is not None and dof_pos_mat is not None:
            # extract the single logged entry with joint names (if present)
            names_list = None
            names_entry = log.get('dof_names_sel', None)
            if names_entry and len(names_entry) > 0:
                first = names_entry[0]
                if isinstance(first, (list, tuple, np.ndarray)):
                    names_list = list(first)

            legs = ['FR', 'FL', 'RR', 'RL']
            ncols = dof_target_mat.shape[1]
            leg_cols = {}
            for leg in legs:
                cols = []
                thigh_name = f'{leg}_thigh_joint'
                calf_name = f'{leg}_calf_joint'
                if names_list is not None:
                    if thigh_name in names_list:
                        cols.append(names_list.index(thigh_name))
                    if calf_name in names_list:
                        cols.append(names_list.index(calf_name))
                # fallback: pick columns that start with leg prefix
                if len(cols) == 0 and names_list is not None:
                    for idx, nm in enumerate(names_list):
                        if nm.startswith(leg + '_') and len(cols) < 2:
                            cols.append(idx)
                cols = [c for c in cols if 0 <= c < ncols]
                leg_cols[leg] = cols

            fig_j, axs_j = plt.subplots(len(legs), 1, figsize=(10, 3 * len(legs)), sharex=True)
            if len(legs) == 1:
                axs_j = [axs_j]
            cmap = plt.get_cmap('tab10')
            for i, leg in enumerate(legs):
                ax = axs_j[i]
                cols = leg_cols.get(leg, [])
                if len(cols) == 0:
                    ax.set_title(f'{leg}: no selected joints')
                    continue
                for k_idx, col in enumerate(cols):
                    color = cmap(k_idx % cmap.N)
                    name = f'joint{col}'
                    if names_list is not None and col < len(names_list):
                        name = names_list[col]
                    ax.plot(time, dof_pos_mat[:, col], color=color, linestyle='-', label=f'{name} (meas)')
                    ax.plot(time, dof_target_mat[:, col], color=color, linestyle='--', label=f'{name} (tgt)')
                ax.set_ylabel('pos [rad]')
                ax.set_title(f'Leg {leg}: measured vs target')
                try:
                    ax.legend(loc='upper right', fontsize='small')
                except Exception:
                    pass
            axs_j[-1].set_xlabel('time [s]')
        else:
            # no joint data
            pass

        # Contact forces: plot per-foot contact z and a binary on-ground indicator
        contact_mat = series_matrix('contact_forces_z')
        fig_c = None
        if contact_mat is not None:
            # sanitize NaNs
            contact_mat = np.nan_to_num(contact_mat)
            nfeet = contact_mat.shape[1]
            # default foot labels if 4 feet (common quadruped order FR, FL, RR, RL)
            if nfeet == 4:
                foot_labels = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
            else:
                foot_labels = [f'foot{i}' for i in range(nfeet)]

            fig_c, axs_c = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            if nfeet == 1:
                axs_c = [axs_c]

            cmap = plt.get_cmap('tab10')
            # top: continuous contact force (Z) per foot
            for j in range(nfeet):
                color = cmap(j % cmap.N)
                axs_c[0].plot(time, contact_mat[:, j], color=color, label=foot_labels[j])
            axs_c[0].set_ylabel('contact force Z [N]')
            axs_c[0].set_title('Contact forces (Z) per foot')
            try:
                axs_c[0].legend(loc='upper right', fontsize='small')
            except Exception:
                pass

            # bottom: binary contact indicator as filled blocks only during contact
            threshold = 1e-3
            offsets = np.arange(0, nfeet * 1.2, 1.2)
            height = 0.9
            for j in range(nfeet):
                binary = (contact_mat[:, j] > threshold)
                color = cmap(j % cmap.N)
                y0 = offsets[j]
                y1 = y0 + height
                # fill_between will only draw where condition is True -> thick blocks for contact
                try:
                    axs_c[1].fill_between(time, y0, y1, where=binary, step='post', color=color, alpha=0.9, linewidth=0)
                except Exception:
                    # fallback: draw thick lines at centers for each contact sample
                    ys = np.where(binary, y0 + 0.5 * height, np.nan)
                    axs_c[1].plot(time, ys, color=color, linewidth=6, solid_capstyle='butt')
            # label rows by foot names
            y_tick_positions = offsets + 0.5 * height
            axs_c[1].set_yticks(y_tick_positions)
            axs_c[1].set_yticklabels(foot_labels)
            axs_c[1].set_xlabel('time [s]')
            axs_c[1].set_title('Contact indicator per foot (filled blocks = on ground)')

        # If no graphical display is available, save figures to files instead
        display_ok = bool(os.environ.get('DISPLAY'))
        try:
            if display_ok:
                plt.tight_layout()
                plt.show()
            else:
                out_dir = os.path.join(os.getcwd(), 'exported_plots')
                os.makedirs(out_dir, exist_ok=True)
                ts = int(time.time())
                main_path = os.path.join(out_dir, f'plots_main_{ts}.png')
                try:
                    fig.savefig(main_path)
                    print(f'Logger: saved main plots to {main_path}')
                except Exception:
                    pass
                # if joint figure was created, save it too
                try:
                    if fig_j is not None:
                        joints_path = os.path.join(out_dir, f'plots_joints_{ts}.png')
                        fig_j.savefig(joints_path)
                        print(f'Logger: saved joint plots to {joints_path}')
                except Exception:
                    pass
                # save contact plots as well
                try:
                    if fig_c is not None:
                        contacts_path = os.path.join(out_dir, f'plots_contacts_{ts}.png')
                        fig_c.savefig(contacts_path)
                        print(f'Logger: saved contact plots to {contacts_path}')
                except Exception:
                    pass
        except Exception:
            try:
                plt.show()
            except Exception:
                pass

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()