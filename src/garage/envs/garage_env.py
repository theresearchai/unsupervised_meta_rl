"""Wrapper class that converts gym.Env into GarageEnv."""

import copy
import time

import numpy as np

import akro
import gym

from garage.envs.env_spec import EnvSpec

# The gym environments using one of the packages in the following lists as
# entry points don't close their viewer windows.
from garage.misc.rllab.box import Box

KNOWN_GYM_NOT_CLOSE_VIEWER = [
    # Please keep alphabetized
    'gym.envs.atari',
    'gym.envs.box2d',
    'gym.envs.classic_control'
]

KNOWN_GYM_NOT_CLOSE_MJ_VIEWER = [
    # Please keep alphabetized
    'gym.envs.mujoco',
    'gym.envs.robotics'
]


class GarageEnv(gym.Wrapper):
    """Returns an abstract Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with a GarageEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, classes inheriting from GarageEnv should silently
    convert action_space and observation_space from gym.Spaces to
    akro.spaces.

    Args:
        env (gym.Env): An env that will be wrapped
        env_name (str): If the env_name is speficied, a gym environment
            with that name will be created. If such an environment does not
            exist, a `gym.error` is thrown.
        is_image (bool): True if observations contain pixel values,
            false otherwise. Setting this to true converts a gym.Spaces.Box
            obs space to an akro.Image and normalizes pixel values.

    """

    def __init__(self, env=None, env_name='', is_image=False):
        # Needed for deserialization
        self._env_name = env_name
        self._env = env

        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)

        if isinstance(self.env.action_space, Box):
            self.action_space = akro.Box(low=self.env.action_space.low, high=self.env.action_space.high)
            self.observation_space = akro.Image(shape=self.env.observation_space.shape)
        else:
            self.action_space = akro.from_gym(self.env.action_space)
            self.observation_space = akro.from_gym(self.env.observation_space,
                                                   is_image=is_image)

        self.__spec = EnvSpec(action_space=self.action_space,
                              observation_space=self.observation_space)

    @property
    def spec(self):
        """Return the environment specification.

        This property needs to exist, since it's defined as a property in
        gym.Wrapper in a way that makes it difficult to overwrite.

        Returns:
            garage.envs.env_spec.EnvSpec: The envionrment specification.

        """
        return self.__spec

    def close(self):
        """Close the wrapped env."""
        self._close_viewer_window()
        self.env.close()

    def _close_viewer_window(self):
        """Close viewer window.

        Unfortunately, some gym environments don't close the viewer windows
        properly, which leads to "out of memory" issues when several of
        these environments are tested one after the other.
        This method searches for the viewer object of type MjViewer, Viewer
        or SimpleImageViewer, based on environment, and if the environment
        is wrapped in other environment classes, it performs depth search
        in those as well.
        This method can be removed once OpenAI solves the issue.
        """
        # We need to do some strange things here to fix-up flaws in gym
        # pylint: disable=import-outside-toplevel
        if self.env.spec:
            if any(package in getattr(self.env.spec, 'entry_point', '')
                   for package in KNOWN_GYM_NOT_CLOSE_MJ_VIEWER):
                # This import is not in the header to avoid a MuJoCo dependency
                # with non-MuJoCo environments that use this base class.
                try:
                    from mujoco_py.mjviewer import MjViewer
                    import glfw
                except ImportError:
                    # If we can't import mujoco_py, we must not have an
                    # instance of a class that we know how to close here.
                    return
                if (hasattr(self.env, 'viewer')
                    and isinstance(self.env.viewer, MjViewer)):
                    glfw.destroy_window(self.env.viewer.window)
            elif any(package in getattr(self.env.spec, 'entry_point', '')
                     for package in KNOWN_GYM_NOT_CLOSE_VIEWER):
                if hasattr(self.env, 'viewer'):
                    from gym.envs.classic_control.rendering import (
                        Viewer, SimpleImageViewer)
                    if (isinstance(self.env.viewer,
                                   (SimpleImageViewer, Viewer))):
                        self.env.viewer.close()

    def reset(self, **kwargs):
        """Call reset on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Args:
            kwargs: Keyword args

        Returns:
            object: The initial observation.

        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """Call step on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Args:
            action (object): An action provided by the agent.

        Returns:
            object: Agent's observation of the current environment
            float : Amount of reward returned after previous action
            bool : Whether the episode has ended, in which case further step()
                calls will return undefined results
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)

        """
        observation, reward, done, info = self.env.step(action)
        # gym envs that are wrapped in TimeLimit wrapper modify
        # the done/termination signal to be true whenever a time
        # limit expiration occurs. The following statement sets
        # the done signal to be True only if caused by an
        # environment termination, and not a time limit
        # termination. The time limit termination signal
        # will be saved inside env_infos as
        # 'GarageEnv.TimeLimitTerminated'
        if 'TimeLimit.truncated' in info:
            info['GarageEnv.TimeLimitTerminated'] = done  # done = True always
            done = not info['TimeLimit.truncated']
        return observation, reward, done, info

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        # the viewer object is not pickleable
        # we first make a copy of the viewer
        env = self.env
        # get the inner env if it is a gym.Wrapper
        if issubclass(env.__class__, gym.Wrapper):
            env = env.unwrapped
        if 'viewer' in env.__dict__:
            _viewer = env.viewer
            # remove the viewer and make a copy of the state
            env.viewer = None
            state = copy.deepcopy(self.__dict__)
            # assign the viewer back to self.__dict__
            env.viewer = _viewer
            # the returned state doesn't have the viewer
            return state
        return self.__dict__

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(state['_env'], state['_env_name'])


class DiaynEnvWrapper(GarageEnv):
    def __init__(self, d, num_s, s, env=None, env_name=''):
        super().__init__(env, env_name)
        self._discriminator = d
        self._num_skills = num_s
        self._skill = s
        self._prob_skills = np.full(self._num_skills, 1.0 / self._num_skills)
        self._prob_skill = 1.0 / self._num_skills

    def step(self, action):
        observation, _, done, info = self.env.step(action)
        return observation, self._obtain_pseudo_reward(observation,
                                                       self._skill), done, info

    def _obtain_pseudo_reward(self, states, skills):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        if isinstance(skills, int):
            skills = np.array([skills])

        q = self._discriminator(states).detach().cpu().detach()
        q_z = np.array([q[i, skills[i]] for i in range(skills.shape[0])])
        reward = np.log(q_z) - np.log(np.full(q_z.shape, self._prob_skill))

        return reward[0]

    def __setstate__(self, state):
        self.__init__(state['_discriminator'], state['_num_skills'],
                      state['_skill'], state['_env'], state['_env_name'])

    def get_training_traj(self, agent):
        env_copy = copy.deepcopy(self)
        pos_traj, _, _, _, _, _ = self._gather_pos_rollout(env_copy, agent)
        return pos_traj

    def _gather_pos_rollout(self,
                            env,
                            agent,  # self.policy
                            max_path_length=np.inf,
                            animated=False,
                            recorded=False,
                            save_video_filename=None,
                            speedup=1,
                            deterministic=False):

        skills = []
        states = []
        actions = []
        self_rewards = []
        env_rewards = []
        agent_infos = []
        env_infos = []
        pos = []
        dones = []

        s = env.reset()
        z = np.eye(self._num_skills)[self._skill]
        agent.reset()
        path_length = 0

        video_buffer = []
        if recorded:
            video_buffer.append(env.render(mode="rgb_array"))

        if animated:
            env.render(mode="human")

        while path_length < (max_path_length or np.inf):
            s = env.observation_space.flatten(s)
            a, agent_info = agent.get_action(s, z)
            if deterministic and 'mean' in agent_infos:
                a = agent_info['mean']
            next_s, env_r, d, env_info = env.step(a)
            self_r = self._obtain_pseudo_reward(s, self._skill)
            states.append(s)
            self_rewards.append(self_r)
            env_rewards.append(env_r)
            actions.append(a)
            skills.append(self._skill)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            dones.append(d)
            pos.append([env.sim.data.qpos[0], env.sim.data.qpos[1]])
            path_length += 1
            if d:
                break
            s = next_s
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
            if recorded:
                video_buffer.append(env.render(mode="rgb_array"))

        if recorded:
            fps = (1 / self._time_per_render)
            self._save_video(video_buffer, save_video_filename, fps)

        return pos, skills, actions, self_rewards, env_rewards, states



