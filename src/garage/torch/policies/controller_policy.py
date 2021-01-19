import torch
from torch import nn
import garage.torch.utils as tu


class ControllerPolicy(nn.Module):
    def __init__(self, controller_policy, num_skills, sub_actor):
        super().__init__()
        self._controller_policy = controller_policy
        self._sub_actor = sub_actor
        self._num_skills = num_skills

    def forward(self, obs):
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        skill_choices, info = self._controller_policy.get_actions(obs)
        mean = info["mean"]
        log_pi = info["log_pi"]
        log_std = info["log_std"]
        skill_dist = info["dist"]
        skill_z = torch.eye(self._num_skills)[skill_choices]
        actions, _ = self._sub_actor.get_actions(obs, skill_z)
        return (actions, mean, log_std, log_pi), skill_dist

    def get_action(self, obs):
        obs = torch.as_tensor(obs[None], device=tu.global_device()).float()
        skill_choice, info = self._controller_policy.get_action(obs)
        skill_z = torch.eye(self._num_skills)[skill_choice]
        action, _ = self._sub_actor.get_action(obs, skill_z)
        return action, skill_choice, info

    @property
    def networks(self):
        return [self._controller_policy, self._sub_actor]
