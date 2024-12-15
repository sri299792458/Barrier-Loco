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

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, data):
        for key, value in data.items():
            if key.startswith('joint_group_'):
                if key not in self.state_log:
                    self.state_log[key] = []
                self.state_log[key].append(value)
            elif isinstance(value, (int, float, np.ndarray)):
                self.log_state(key, value)
            else:
                print(f"Warning: Unexpected data type for key {key}: {type(value)}")


    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    # def plot_states(self):
    #     # self.plot_process = Process(target=self._plot)
    #     # self.plot_process.start()

    # def _plot(self):
    #     fig = plt.figure(1, figsize=(15, 25)) 
    #     axs = fig.subplots(5, 2)
        
    #     for key, value in self.state_log.items():
    #         if not isinstance(value[0], dict):
    #             time = np.linspace(0, len(value)*self.dt, len(value))
    #             break
    #     log = self.state_log

    #     # Plot base velocities
    #     self._plot_base_velocities(axs[0, 0], axs[0, 1], time, log)

    #     # Plot joint groups
    #     for group in range(4):
    #         self._plot_joint_group(axs[group+1, 0], axs[group+1, 1], time, log, group)

    #     plt.tight_layout()
    #     plt.show()
    def _plot_base_velocities(self, ax1, ax2, time, log):
        # Plot base vel x and y
        for ax, vel_key, cmd_key, title in [
            (ax1, "base_vel_x", "command_x", "Base velocity x"),
            (ax2, "base_vel_y", "command_y", "Base velocity y")
        ]:
            if log[vel_key]: ax.plot(time, log[vel_key], label='measured')
            if log[cmd_key]: ax.plot(time, log[cmd_key], label='commanded')
            ax.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title=title)
            ax.legend()

    def _plot_joint_group(self, ax_pos, ax_torque, time, log, group):
        group_data = log[f'joint_group_{group}']
        
        # Plot positions
        for j in range(3):
            ax_pos.plot(time, [data[f'dof_pos_{j}'] for data in group_data], label=f'Joint {group*3 + j}')
        ax_pos.set_title(f'Joint Positions Group {group}')
        ax_pos.legend()
        ax_pos.set_xlabel('Time [s]')
        ax_pos.set_ylabel('Position [rad]')
        
        # Plot torques
        for j in range(3):
            ax_torque.plot(time, [data[f'dof_torque_{j}'] for data in group_data], label=f'Joint {group*3 + j}')
        ax_torque.set_title(f'Joint Torques Group {group}')
        ax_torque.legend()
        ax_torque.set_xlabel('Time [s]')
        ax_torque.set_ylabel('Torque [Nm]')

    # def plot_height_data(self):
    #     self.height_plot_process = Process(target=self._plot_height)
    #     self.height_plot_process.start()

    # def _plot_height(self):
    #     fig = plt.figure(2, figsize=(10, 8))  # Specify figure 2
    #     axs = fig.subplots(2, 1)
        
    #     # Get time array
    #     for key, value in self.state_log.items():
    #         if not isinstance(value[0], dict):
    #             time = np.linspace(0, len(value)*self.dt, len(value))
    #             break
        
    #     log = self.state_log
        
    #     # Plot feet z positions
    #     for foot in range(4):
    #         foot_data = log[f'foot_{foot}_pos']
    #         z_positions = [pos['z'] for pos in foot_data]
    #         ax1.plot(time, z_positions, label=f'Foot {foot}')
        
    #     ax1.set_title('Feet Z Positions')
    #     ax1.set_xlabel('Time [s]')
    #     ax1.set_ylabel('Height [m]')
    #     ax1.legend()
        
    #     # Plot base height
    #     if 'base_height' in log:
    #         ax2.plot(time, log['base_height'], label='Base Height')
    #     ax2.set_title('Base Height')
    #     ax2.set_xlabel('Time [s]')
    #     ax2.set_ylabel('Height [m]')
    #     ax2.legend()
        
    #     plt.tight_layout()
    #     plt.show()
    def plot_states(self):
    # Remove process creation, call directly
        self._plot()

    def _plot(self):
        fig = plt.figure(1, figsize=(15, 25))
        axs = fig.subplots(5, 2)
        
        for key, value in self.state_log.items():
            if not isinstance(value[0], dict):
                time = np.linspace(0, len(value)*self.dt, len(value))
                break
        log = self.state_log

        self._plot_base_velocities(axs[0, 0], axs[0, 1], time, log)

        for group in range(4):
            self._plot_joint_group(axs[group+1, 0], axs[group+1, 1], time, log, group)

        plt.figure(1)  # Ensure figure 1 is active
        plt.show(block=False)  # Don't block

    def plot_height_data(self):
        # Remove process creation, call directly
        self._plot_height()

    def _plot_height(self):
        fig = plt.figure(2, figsize=(10, 8))
        axs = fig.subplots(2, 1)
        
        log = self.state_log
        
         # Plot feet z positions
        if 'feet_pos_z' in log and len(log['feet_pos_z']) > 0:
            feet_data = np.array(log['feet_pos_z'])  # Convert list to numpy array
            time = np.linspace(0, len(feet_data)*self.dt, len(feet_data))
            for foot in range(4):
                axs[0].plot(time, feet_data[:, foot], label=f'Foot {foot}')
            
            axs[0].set_title('Feet Z Positions')
            axs[0].set_xlabel('Time [s]')
            axs[0].set_ylabel('Height [m]')
            axs[0].legend()
        else:
            axs[0].text(0.5, 0.5, 'No foot position data available', 
                    horizontalalignment='center', verticalalignment='center')
    
        
     
        
        # Plot base height
        if 'base_height' in log and log['base_height']:
            time = np.linspace(0, len(log['base_height'])*self.dt, len(log['base_height']))
            axs[1].plot(time, log['base_height'], label='Base Height')
            axs[1].set_title('Base Height')
            axs[1].set_xlabel('Time [s]')
            axs[1].set_ylabel('Height [m]')
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'No base height data available', 
                        horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.figure(2)
        plt.show(block=True)
    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()