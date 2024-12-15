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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage

class HIMPPO:
    actor_critic: HIMActorCritic
    def __init__(self,
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
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = {
            "standard": self.actor_critic.evaluate(critic_obs, critic_type="standard").detach(),
            "barrier": self.actor_critic.evaluate(critic_obs, critic_type="barrier").detach(),
        }      
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = {
            "standard": rewards["standard"].clone(),
            "barrier": rewards["barrier"].clone(),
        }
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards["standard"] += self.gamma * torch.squeeze(
                self.transition.values["standard"] * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
            self.transition.rewards["barrier"] += self.gamma * torch.squeeze(
                self.transition.values["barrier"] * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values = {
            "standard": self.actor_critic.evaluate(last_critic_obs, critic_type="standard").detach(),
            "barrier": self.actor_critic.evaluate(last_critic_obs, critic_type="barrier").detach(),
        }
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        # mean_value_loss = 0
        mean_value_loss_standard = 0   
        mean_value_loss_barrier = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_standard_batch, target_values_barrier_batch, advantages_standard_batch, advantages_barrier_batch, returns_standard_batch, returns_barrier_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                # value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                value_standard = self.actor_critic.evaluate(critic_obs_batch, critic_type="standard")
                value_barrier = self.actor_critic.evaluate(critic_obs_batch, critic_type="barrier")
               
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                # estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)
                estimation_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch[:,47:50], lr=self.learning_rate)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
               
                # Separate surrogate losses for standard and barrier advantages
                surrogate_standard = -torch.squeeze(advantages_standard_batch) * ratio
                surrogate_standard_clipped = -torch.squeeze(advantages_standard_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss_standard = torch.max(surrogate_standard, surrogate_standard_clipped).mean()

                surrogate_barrier = -torch.squeeze(advantages_barrier_batch) * ratio
                surrogate_barrier_clipped = -torch.squeeze(advantages_barrier_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss_barrier = torch.max(surrogate_barrier, surrogate_barrier_clipped).mean()
                surrogate_loss = 0.5 * (surrogate_loss_standard + surrogate_loss_barrier)
                # surrogate_loss = surrogate_loss_standard

                # Value function loss (separate for standard and barrier)
                if self.use_clipped_value_loss:
                    value_standard_clipped = target_values_standard_batch + (
                        value_standard - target_values_standard_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_barrier_clipped = target_values_barrier_batch + (
                        value_barrier - target_values_barrier_batch
                    ).clamp(-self.clip_param, self.clip_param)

                    value_losses_standard = (value_standard - returns_standard_batch).pow(2)
                    value_losses_standard_clipped = (value_standard_clipped - returns_standard_batch).pow(2)
                    value_loss_standard = torch.max(value_losses_standard, value_losses_standard_clipped).mean()

                    value_losses_barrier = (value_barrier - returns_barrier_batch).pow(2)
                    value_losses_barrier_clipped = (value_barrier_clipped - returns_barrier_batch).pow(2)
                    value_loss_barrier = torch.max(value_losses_barrier, value_losses_barrier_clipped).mean()
                else:
                    value_loss_standard = (returns_standard_batch - value_standard).pow(2).mean()
                    value_loss_barrier = (returns_barrier_batch - value_barrier).pow(2).mean()

                # Combine value losses
                value_loss = 0.5 * (value_loss_standard + value_loss_barrier)

              

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # mean_value_loss += value_loss.item()
                mean_value_loss_standard += value_loss_standard.item()
                mean_value_loss_barrier += value_loss_barrier.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                # mean_swap_loss += swap_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss_standard /= num_updates
        mean_value_loss_barrier /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        # mean_swap_loss /= num_updates
        self.storage.clear()

        # return mean_value_loss_standard, mean_value_loss_barrier, mean_surrogate_loss, estimation_loss, swap_loss
        return mean_value_loss_standard, mean_value_loss_barrier, mean_surrogate_loss, estimation_loss
