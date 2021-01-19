import numpy as np
import torch

import garage.torch.utils as tu
from garage.torch.modules import MLPModule


class MLPDiscriminator(MLPModule):
    def __init__(self, env_spec, skills_num, **kwargs):
        self._obs_dim = env_spec.observation_space.flat_dim
        self._skills_num = skills_num

        super().__init__(input_dim=self._obs_dim,
                         output_dim=skills_num,
                         **kwargs)

    def forward(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float().to(
                tu.global_device())
        # print("in forward")
        # print(states.size())
        # states = torch.from_numpy(np.array([1, 2, 3])).float().to(tu.global_device())
        x = super().forward(states)
        return torch.softmax(x, dim=-1)

    def infer_skills(self, states):
        with torch.no_grad():
            # if not isinstance(states, torch.Tensor):
            #    states = torch.from_numpy(states).float().to(
            #        tu.global_device())
            # print("in infer_skills")
            # print(states.size())
            x = self.forward(torch.Tensor(states))
            return np.array([np.random.choice(self._skills_num, p=x.numpy()[idx])
                             for idx in range(x.numpy().shape[0])])

    def infer_skill(self, state):
        with torch.no_grad():
            # if not isinstance(state, torch.Tensor):
            #    state = torch.from_numpy(state).float().to(
            #        tu.global_device())
            # print("in infer_skill")
            # print(state.size())
            x = self.forward(torch.Tensor(state).unsqueeze(0))
            return np.random.choice(self._skills_num, p=x.squeeze(0).numpy())
