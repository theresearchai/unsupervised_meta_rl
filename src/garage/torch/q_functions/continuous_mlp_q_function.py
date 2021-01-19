"""This modules creates a continuous Q-function network."""

import torch

from garage.torch.modules import MLPModule
import garage.torch.utils as tu


class ContinuousMLPQFunction(MLPModule):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim + self._action_dim,
                           output_dim=1,
                           **kwargs)

    def forward(self, observations, actions):
        """Return Q-value(s)."""
        return super().forward(torch.cat([observations, actions], 1))

class ContinuousMLPSkillQFunction(MLPModule):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, skills_num, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._skill_dim = skills_num

        MLPModule.__init__(self,
                           input_dim=self._obs_dim + self._action_dim + self._skill_dim,
                           output_dim=1,
                           **kwargs)

    def forward(self, states, actions, skills):
        """Return Q-value(s)."""
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float().to(
                tu.global_device())
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions).float().to(
                tu.global_device())
        if not isinstance(skills, torch.Tensor):
            skills = torch.from_numpy(skills).float().to(
                tu.global_device())

        return super().forward(torch.cat([states, skills, actions], 1))

    # def forward(self, observations, actions):
    #     """Return Q-value(s)."""
    #     if not isinstance(actions, torch.Tensor):
    #         actions = torch.from_numpy(actions).float().to(
    #             tu.global_device())

    #     return super().forward(torch.cat([observations, actions], 1))
