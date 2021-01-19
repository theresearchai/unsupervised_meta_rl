

import numpy as np

from garage.misc.rllab import spaces
from garage.misc.rllab.env_spec import EnvSpec
from garage.misc.rllab.serializable import Serializable


class MetaEnv(Serializable):
    def __init__(self, env, base_policy, num_skills, steps_per_option=100):
        Serializable.quick_init(self, locals())
        self._base_policy = base_policy
        self._env = env
        self._steps_per_option = steps_per_option
        self._num_skills = num_skills
        self.observation_space = self._env.observation_space
        self.action_space = spaces.Discrete(num_skills)
        self.spec = EnvSpec(self.observation_space, self.action_space)
        self._obs = self.reset()

    def step(self, meta_action):
        total_reward = 0
        for _ in range(self._steps_per_option):
            aug_obs = concat_obs_z(self._obs, meta_action, self._num_skills)
            (action, _) = self._base_policy.get_action(aug_obs)
            (self._obs, r, done, _) = self._env.step(action)
            total_reward += r
            if done: break
        # Normalize the total reward by number of steps
        return (self._obs, total_reward / float(self._steps_per_option), done, {})

    def reset(self):
        return self._env.reset()

    def log_diagnostics(self, paths):
        self._env.log_diagnostics(paths)

    def terminate(self):
        self._env.terminate()


class FixedOptionEnv(Serializable):
    def __init__(self, env, num_skills, z):
        Serializable.quick_init(self, locals())
        self._env = env
        self._num_skills = num_skills
        self._z = z
        obs_space = self._env.observation_space
        low = np.hstack([obs_space.low, np.full(num_skills, 0)])
        high = np.hstack([obs_space.high, np.full(num_skills, 1)])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = self._env.action_space
        self.spec = EnvSpec(self.observation_space, self.action_space)

    def step(self, action):
        (obs, r, done, info) = self._env.step(action)
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return (aug_obs, r, done, info)

    def reset(self):
        obs = self._env.reset()
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return aug_obs

    def log_diagnostics(self, paths):
        self._env.log_diagnostics(paths)

    def terminate(self):
        self._env.terminate()

    def concat_obs_z(obs, z, num_skills):
        """Concatenates the observation to a one-hot encoding of Z."""
        assert np.isscalar(z)
        z_one_hot = np.zeros(num_skills)
        z_one_hot[z] = 1
        return np.hstack([obs, z_one_hot])
