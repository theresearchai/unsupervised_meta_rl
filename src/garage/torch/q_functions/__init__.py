"""PyTorch Q-functions."""
from garage.torch.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from garage.torch.q_functions.continuous_mlp_q_function import ContinuousMLPSkillQFunction

__all__ = ['ContinuousMLPQFunction',
           'ContinuousMLPSkillQFunction']
