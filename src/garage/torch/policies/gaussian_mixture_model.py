import torch
import numpy as np

import garage.torch.utils as tu
from garage.torch.distributions.gmm import GMM
from garage.torch.policies.policy import Policy

EPS = 1e-6

class GMMSkillPolicy(Policy, torch.nn.Module):
    def __init__(self, env_spec, K=2, hidden_layer_sizes=(256, 256), skills_num=0,
                 reg=1e-3, squash=True, reparameterize=False, qf=None,
                 name="GaussianMixtureModel"):
        self._skills_num = skills_num
        self._hidden_layers = hidden_layer_sizes
        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim + skills_num
        self._K = K
        self._is_deterministic = False
        self._fixed_h = None
        self._squash = squash
        self._qf = qf
        self._reg = reg

        assert not reparameterize
        self._reparameterize = reparameterize

        self.distribution = GMM(K=self._K,
                                hidden_layer_sizes=self._hidden_layers,
                                Dx=self._Da,
                                mlp_input_dim=self._Ds,
                                reg=self._reg)

        Policy.__init__(self, env_spec, name)
        torch.nn.Module.__init__(self)

    #def get_action(self, observation):
    #    return self.get_actions(observation[None])[0]

    def get_action(self, observation, skill):
        return self.get_actions(observation, skill)

    def forward(self, observations, skills):
        if not isinstance(observations, torch.Tensor):
            observations = torch.from_numpy(observations).float().to(
                tu.global_device())
            if len(observations.shape) == 1:
                observations = observations.unsqueeze(0)
        if not isinstance(skills, torch.Tensor):
            skills = torch.from_numpy(skills).float().to(
                tu.global_device())
            if len(skills.shape) == 1:
                skills = skills.unsqueeze(0)
        input = torch.cat([observations, skills], dim=1).to(tu.global_device())

        log_p_x_t, reg_loss_t, x_t, log_ws_t, mus_t, log_sigs_t = self.distribution.get_p_params(
            input)
        raw_actions = x_t.detach().cpu().numpy()
        actions = np.tanh(raw_actions) if self._squash else raw_actions

        return actions, dict(log_p_x_t=log_p_x_t, reg_loss_t=reg_loss_t,
                             x_t=x_t, log_ws_t=log_ws_t, mus_t=mus_t,
                             log_sigs_t=log_sigs_t)

    def get_actions(self, observations, skills):
        return self.forward(observations, skills)

    def _squash_correction(self, actions):
        if not self._squash:
            return 0
        else:
            return np.sum(np.log(1-np.tanh(actions) ** 2 + EPS), axis=1)

    @property
    def parameters(self, recurse=True):
        return self.distribution.parameters

    @property
    def networks(self):
        return self.distribution.networks

    def reset(self):
        pass







