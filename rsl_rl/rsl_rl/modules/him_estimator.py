import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class HIMEstimator(nn.Module):
    def __init__(self, temporal_steps, num_one_step_obs, learning_rate=1e-3, max_grad_norm=10.0):
        """
        Predicts the velocity of the robot body from observation history using supervised learning.
        
        Args:
        - temporal_steps: Number of time steps in the observation history.
        - num_one_step_obs: Number of features per observation at one time step.
        - learning_rate: Learning rate for the optimizer.
        - max_grad_norm: Maximum gradient norm for clipping.
        """
        super(HIMEstimator, self).__init__()
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.max_grad_norm = max_grad_norm

        # Calculate input dimension as temporal_steps * num_one_step_obs
        input_dim = temporal_steps * num_one_step_obs

        # Define MLP architecture with [256, 128] hidden layers and ELU activations
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 3)  # Output dimension is 3 for velocity prediction
        )

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs_history):
        """
        Forward pass to predict velocity from observation history.
        
        Args:
        - obs_history: Tensor of shape (batch_size, input_dim).
        
        Returns:
        - pred_vel: Predicted velocity tensor of shape (batch_size, 3).
        """
        pred_vel = self.network(obs_history)
        return pred_vel

    def update(self, obs_history, true_vel, lr=None):
        """
        Performs a single training step to minimize the supervised loss for velocity prediction.
        
        Args:
        - obs_history: Observation history input tensor (batch_size, input_dim).
        - true_vel: Ground truth velocity tensor (batch_size, 3).
        - lr: Optional learning rate to update during training.

        Returns:
        - loss: The MSE loss for the prediction.
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        # Forward pass
        pred_vel = self.forward(obs_history)

        # Compute mean squared error loss
        loss = F.mse_loss(pred_vel, true_vel)

        # Backpropagation and parameter update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


def get_activation(act_name):
    """
    Utility function to get activation function by name.
    """
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("Invalid activation function!")
        return None


