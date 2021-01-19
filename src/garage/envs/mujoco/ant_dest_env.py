"""Variant of the AntEnv with different target velocity."""
import numpy as np
import math

from garage.envs.mujoco.ant_env_meta_base import AntEnvMetaBase  # noqa: E501


class AntVelEnv(AntEnvMetaBase):

    def __init__(self, task=None):
        super().__init__(task or {'velocity': 0.})

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
                    * task_dest (float): Target destination.
                        Usually between (-5, 5) to (5, 5)

        """
        xposbefore = self.sim.data.qpos[0]
        yposbefore = self.sim.data.qpos[1]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        yposafter = self.sim.data.qpos[1]

        dest_x = self._task['destination'][0]
        dest_y = self._task['destination'][1]

        distbefore = math.sqrt((dest_x - xposbefore)**2 + (dest_y - yposbefore)**2)
        distafter = math.sqrt((dest_x - xposafter)**2 + (dest_y - yposafter)**2)
        forward_reward = distbefore - distafter
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task_dest=self._task['destination'])
        return observation, reward, done, infos

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, float]]: A list of "tasks," where each task is a
                dictionary containing a single key, "destination", mapping to a
                list with x and y value between -5 and 5.

        """
        destinations = [self.np_random.uniform(-5, 5, size=(2, )) for _ in range(num_tasks)]
        tasks = [{'destination': destination} for destination in destinations]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, [float, float]]): A task (a dictionary containing a single
                key, "destination", and a list that contains x and y, generally between
                -5 to 5).

        """
        self._task = task

    def get_task(self):
        return self._task
