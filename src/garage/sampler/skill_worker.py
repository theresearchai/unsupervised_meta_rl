"""Skill Worker class."""
import sys
from collections import defaultdict

import numpy as np

from garage import SkillTrajectoryBatch
from garage.sampler import DefaultWorker


class SkillWorker(DefaultWorker):
    def __init__(
                self,
                *,
                seed,
                max_path_length,
                worker_number,
                skills_num):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)

        self._prob_skill = np.full(skills_num, 1.0 / skills_num)
        self._skills_num = skills_num
        self._skills = []
        self._states = []
        self._last_states = []
        self._cur_z = None
        self._prev_s = None

    # def worker_init(self):

    # def update_agent(self, agent_update):

    # def update_env(self, env_update):

    def start_rollout(self, skill=None):
        self._path_length = 0
        self._prev_s = self.env.reset()
        if skill is None:
            self._cur_z = self._sample_skill()
        else:
            self._cur_z = skill
        self.agent.reset()

    def step_rollout(self):
        if self._path_length < self._max_path_length:
            z_onehot = np.eye(self._skills_num)[self._cur_z]
            a, agent_info = self.agent.get_action(self._prev_s, z_onehot)
            next_s, r, d, env_info = self.env.step(a)
            self._states.append(self._prev_s)
            self._rewards.append(r)
            self._actions.append(a)
            self._skills.append(self._cur_z)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)
            np.set_printoptions(threshold=sys.maxsize)
            # print("action")
            # a
            if not d:
                self._prev_s = next_s
                return False
        self._lengths.append(self._path_length)
        self._last_states.append(self._prev_s)
        return True

    def collect_rollout(self):
        states = self._states
        self._states = []
        last_states = self._last_states
        self._last_states = []
        skills = self._skills
        self._skills = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        terminals = self._terminals
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []

        return SkillTrajectoryBatch(env_spec=self.env.spec,
                                    num_skills=self._skills_num,
                                    skills=np.asarray(skills),
                                    states=np.asarray(states),
                                    last_states=np.asarray(last_states),
                                    actions=np.asarray(actions),
                                    env_rewards=np.asarray(rewards),  # env_rewards
                                    terminals=np.asarray(terminals),
                                    env_infos=dict(env_infos),
                                    agent_infos=dict(agent_infos),
                                    lengths=np.asarray(lengths, dtype='i'))

    def rollout(self, skill=None):
        self.start_rollout(skill)
        # print("rollout started")
        while not self.step_rollout():
            # print("rollout step={}".format(step))
            # step = step+1
            pass
        return self.collect_rollout()

    # def shutdown(self)

    def _sample_skill(self):  # uniform dist. in order to maximize entropy
        return np.random.choice(self._skills_num, p=self._prob_skill)
