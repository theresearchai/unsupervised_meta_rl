"""Variant of the AntEnv with different target directions."""
import numpy as np

from garage.envs.mujoco.ant_env_meta_base import AntEnvMetaBase  # noqa: E501


class AntDirEnv(AntEnvMetaBase):

    def __init__(self, task=None):
        super().__init__(task or {'direction': 1.})

    def step(self, action):
        """Take one step in the environment.

        Equivalent to step in HalfCheetahEnv, but with different rewards.

        Args:
            action (np.ndarray): The action to take in the environment.

        Returns:
            tuple:
                * observation (np.ndarray): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step. Always False for this environment.
                * infos (dict):
                    * reward_forward (float): Reward for moving, ignoring the
                        control cost.
                    * reward_ctrl (float): The reward for acting i.e. the
                        control cost (always negative).
                    * task_dir (float): Target direction. 1.0 for forwards,
                        -1.0 for backwards.

        """
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._task['direction'] * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task_dir=self._task['direction'])
        return observation, reward, done, infos

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, float]]: A list of "tasks," where each task is a
                dictionary containing a single key, "direction", mapping to -1
                or 1.

        """
        directions = (
            2 * self.np_random.binomial(1, p=0.5, size=(num_tasks, )) - 1)
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, float]): A task (a dictionary containing a single
                key, "direction", mapping to -1 or 1).

        """
        self._task = task

    def get_task(self):
        return self._task
