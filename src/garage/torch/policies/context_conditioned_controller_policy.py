import abc

import numpy as np
import torch
import torch.nn.functional as F

import garage.torch.utils as tu
from garage.torch.policies import ContextConditionedPolicy


class OpenContextConditionedControllerPolicy(ContextConditionedPolicy):
    """

    """
    def __init__(self, latent_dim, context_encoder, controller_policy,
                 num_skills, sub_actor, use_information_bottleneck,
                 use_next_obs):
        super().__init__(latent_dim, context_encoder, controller_policy,
                         use_information_bottleneck, use_next_obs)
        self._policy = None  # for naming consistency
        self._controller_policy = controller_policy
        self._sub_actor = sub_actor
        self._num_skills = num_skills

    # def reset_belief(self, num_tasks=1):
    # resets self.z_means, self.z_vars, self._context, self._context_encoder

    def update_context(self, timestep):
        # append a single timestep [o, a, r, no] to the context
        '''
        [[[o1, o2, o3, .., a1, a2, ...], -> timestep = 1
        [o1, o2, o3, .., a1, a2, ...]]] -> timestep = 2
        '''
        o = torch.as_tensor(timestep.state[None, None, ...],
                            device=tu.global_device()).float()
        a = torch.as_tensor(timestep.action[None, None, ...],
                            device=tu.global_device()).float()
        r = torch.as_tensor(np.array([timestep.env_reward])[None, None, ...],
                            device=tu.global_device()).float()
        s = torch.as_tensor(np.array([timestep.skill])[None, None, ...],
                            device=tu.global_device()).float()
        no = torch.as_tensor(timestep.next_state[None, None, ...],
                             device=tu.global_device()).float()

        if self._use_next_obs:
            data = torch.cat([o, a, r, s, no], dim=2)
        else:
            data = torch.cat([o, a, r, s], dim=2)

        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context): # need to write a more flexible way - to get z_means, z_vars
        '''
            Compute q(z|c)
            context (X, N, C): X is the number of tasks, N is batch sizes,
            C is the combined size of o, a, r, no if used
        '''
        if self._use_information_bottleneck:
            self.z_means, self.z_vars = self._context_encoder.infer_posterior(context)
        else:
            self.z_means = self._context_encoder.infer_posterior(context)
        self.z_vars[self.z_vars != self.z_vars] = 0
        self.z_means[self.z_means != self.z_means] = 0
        self.sample_from_belief()

    # def sample_from_belief(self):
    # with dist(z_mean, z_var), sample z if use info_bottleneck
    # if noy, z = z_mean

    def forward(self, obs, context):
        self.infer_posterior(context)
        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        obs_z = torch.cat([obs, task_z.detach()], dim=1)
        skill_choices, info = self._controller_policy.get_actions(obs_z)
        mean = info["mean"]
        log_pi = info["log_pi"]
        log_std = info["log_std"]
        skill_dist = info["dist"]

        skill_z = torch.eye(self._num_skills)[skill_choices]  # np or torch
        actions, _ = self._sub_actor.get_actions(obs, skill_z)

        return (actions, mean, log_std, log_pi), skill_dist, task_z

    def get_action(self, obs):
        z = self.z
        obs = torch.as_tensor(obs[None], device=tu.global_device()).float()
        obs_in = torch.cat([obs, z], dim=1)
        skill_choice, info = self._controller_policy.get_action(obs_in)
        skill_z = torch.eye(self._num_skills)[skill_choice]
        action, _ = self._sub_actor.get_action(obs, skill_z)
        return action, skill_choice, info

    # embed obs with cat
    # sample action from policy - similar to forward()

    # def compute_kl_div(self):
    # compute KL(q(z|c)|p(z))

    @property
    def networks(self):
        return [self._context_encoder.network, self._controller_policy,
                self._sub_actor]

    # @property
    # def context(self):

class ContextEncoder(abc.ABC):

    @abc.abstractmethod
    def infer_posterior(self, context):
        ''' returns z_means, z_vars'''

    @abc.abstractmethod
    def reset(self):
        '''resets'''

# process a sequence of contexts
class GaussianContextEncoder(ContextEncoder):

    def __init__(self, context_encoder, use_information_bottleneck, latent_dim):
        self._context_encoder = context_encoder
        self._use_information_bottleneck = use_information_bottleneck
        self._latent_dim = latent_dim

    def infer_posterior(self, context):
        # print(context.size())
        params = self._context_encoder.forward(context)
        params = params.view(context.size(0), -1,
                             self._context_encoder.output_dim)

        # with probabilistic z, predict mean and variance of q(z | c)
        if self._use_information_bottleneck:
            mu = params[..., :self._latent_dim]
            sigma_squared = F.softplus(params[..., self._latent_dim:])
            z_params = [
                tu.product_of_gaussians(m, s)
                for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
            ]
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
            return z_means, z_vars
        else:
            z_means = torch.mean(params, dim=1)
            return z_means

    def reset(self):
        self._context_encoder.reset()

    @property
    def network(self):
        return self._context_encoder

# should be updated with controller policies for once a while
class LSTMContextEncoder(ContextEncoder):
    pass


