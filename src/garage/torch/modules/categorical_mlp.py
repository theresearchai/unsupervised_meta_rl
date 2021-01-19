import numpy as np
import torch

import garage.torch.utils as tu
from garage.torch.modules import MLPModule
from garage.torch.policies import Policy


class CategoricalMLPPolicy(Policy, MLPModule):
    def __init__(self, env_spec, name="CategoricalMLPPolicy", **kwargs):
        self._obs_dim = env_spec.input_space.flat_dim
        self._action_dim = env_spec.output_space.flat_dim

        Policy.__init__(self, env_spec, name)
        MLPModule.__init__(self, input_dim=self._obs_dim,
                           output_dim=self._action_dim,
                           **kwargs)

    def forward(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.from_numpy(states).float().to(
                tu.global_device())
        dist = super().forward(states)
        return torch.softmax(dist, dim=-1)

    def get_actions(self, states):
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.from_numpy(states).float().to(
                    tu.global_device())
            states = states.to(tu.global_device())
            dist = self.forward(states).to('cpu').detach()
            actions = np.array([np.random.choice(self._action_dim,
                                                 p=dist.numpy()[idx])
                                for idx in range(dist.numpy().shape[0])])
            ret_mean = np.mean(dist.numpy())
            ret_log_std = np.log((np.std(dist.numpy())))

            len = actions.shape[0]
            ret_log_pi = np.log(dist[np.arange(len), actions])

            return (actions, dict(mean=ret_mean, log_std=ret_log_std,
                                  log_pi=ret_log_pi, dist=dist))

    def get_action(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float().to(
                    tu.global_device())

            state = state.to(tu.global_device())
            dist = self.forward(state.unsqueeze(0)).squeeze(0).to('cpu').detach()

            # action = torch.tensor([torch.rsample()])
            action = np.array([np.random.choice(self._action_dim,
                                                p=dist.squeeze(0).numpy())])

            ret_mean = np.mean(dist.numpy())
            ret_log_std = np.log((np.std(dist.numpy())))
            ret_log_pi = np.log(dist[..., list(action)])

            return (action, dict(mean=ret_mean, log_std=ret_log_std,
                                 log_pi=ret_log_pi, dist=dist))

