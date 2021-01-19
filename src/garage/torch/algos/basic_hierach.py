import copy
import sys
from collections import defaultdict

import akro
import numpy as np
import torch
import torch.nn.functional as F
from dowel import logger

import garage.torch.utils as tu
from garage import SkillTrajectoryBatch, InOutSpec, SkillTimeStep
from garage.envs import EnvSpec
from garage.experiment import MetaEvaluator
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch.policies.controller_policy import ControllerPolicy


class MetaBasicHierch(MetaRLAlgorithm):
    def __init__(self,
                 env,
                 skill_env,
                 controller_policy,
                 skill_actor,
                 qf,
                 vf,
                 num_skills,
                 num_train_tasks,
                 num_test_tasks,
                 test_env_sampler,
                 sampler_class,  # to avoid cycling import
                 controller_class=ControllerPolicy,
                 controller_lr=3E-4,
                 qf_lr=3E-4,
                 vf_lr=3E-4,
                 policy_mean_reg_coeff=1E-3,
                 policy_std_reg_coeff=1E-3,
                 policy_pre_activation_coeff=0.,
                 soft_target_tau=0.005,
                 optimizer_class=torch.optim.Adam,
                 meta_batch_size=64,
                 num_steps_per_epoch=1000,
                 num_tasks_sample=5,
                 batch_size=1024,
                 embedding_batch_size=1024,
                 embedding_mini_batch_size=1024,
                 max_path_length=1000,
                 discount=0.99,
                 replay_buffer_size=1000000,
                 ):

        self._env = env
        self._skill_env = skill_env
        self._qf1 = qf
        self._qf2 = copy.deepcopy(qf)
        self._vf = vf
        self._skill_actor = skill_actor
        self._num_skills = num_skills
        self._num_train_tasks = num_train_tasks
        self._num_test_tasks = num_test_tasks

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau

        self._meta_batch_size = meta_batch_size
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_tasks_sample = num_tasks_sample
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size

        self._task_idx = None
        self._skill_idx = None  # do we really need it

        self._is_resuming = False

        worker_args = dict(num_skills=num_skills,
                           skill_actor_class=type(skill_actor),
                           controller_class=controller_class,
                           deterministic=False, accum_context=False)
        self._evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                        max_path_length=max_path_length,
                                        worker_class=BasicHierachWorker,
                                        worker_args=worker_args,
                                        n_test_tasks=num_test_tasks,
                                        sampler_class=sampler_class,
                                        trajectory_batch_class=SkillTrajectoryBatch)
        self._average_rewards = []

        self._controller = controller_class(
            controller_policy=controller_policy,
            num_skills=num_skills,
            sub_actor=skill_actor)

        self._skills_replay_buffer = PathBuffer(replay_buffer_size)

        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self.target_vf = copy.deepcopy(self._vf)
        self.vf_criterion = torch.nn.MSELoss()

        self._controller_optimizer = optimizer_class(
            self._controller.networks[1].parameters(),
            lr=controller_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self._qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self._qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self._vf.parameters(),
            lr=vf_lr,
        )

    def train(self, runner):
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self._num_steps_per_epoch

            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self._obtain_task_samples(runner, epoch,
                                              self._num_initial_steps, np.inf)
                self._is_resuming = False

            logger.log('Sampling tasks')
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._replay_buffers[idx].clear()
                self._obtain_task_samples(runner, epoch, self._num_steps_per_epoch)

            logger.log('Training task adapting...')
            self._tasks_adapt_train_once()

            runner.step_itr += 1

            logger.log('Evaluating...')
            # evaluate
            self._controller.reset_belief()
            self._average_rewards.append(self._evaluator.evaluate(self))

        return self._average_rewards

    def _tasks_adapt_train_once(self):
        for _ in range(self._num_steps_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
            self._tasks_adapt_optimize_policy(indices)

    def _tasks_adapt_optimize_policy(self, indices):
        num_tasks = len(indices)
        obs, actions, rewards, skills, next_obs, terms = self._sample_task_path(indices)
        self._controller.reset_belief(num_tasks=num_tasks)

        # data shape is (task, batch, feat)
        # new_skills_pred is distribution
        policy_outputs, new_skills_pred, task_z = self._controller(obs)
        new_actions, policy_mean, policy_log_std, policy_log_pi = policy_outputs[:4]

        # flatten out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        # actions = actions.view(t * b, -1)
        skills = skills.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize qf and encoder networks
        # TODO try [obs, skills, actions] or [obs, skills, task_z]
        # FIXME prob need to reshape or tile task_z
        obs = obs.to(tu.global_device())
        skills = skills.to(tu.global_device())
        next_obs = next_obs.to(tu.global_device())

        q1_pred = self._qf1(torch.cat([obs, skills], dim=1))
        q2_pred = self._qf2(torch.cat([obs, skills], dim=1))

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        rewards_flat = rewards.view(b * t, -1)
        terms_flat = terms.view(b * t, -1)
        q_target = rewards_flat + (1. - terms_flat) * self._discount
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean(
            (q2_pred - q_target) ** 2)
        qf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        new_skills_pred = new_skills_pred.to(tu.global_device())

        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([obs, new_skills_pred], dim=1),
                       task_z.detach())
        q2 = self._qf2(torch.cat([obs, new_skills_pred], dim=1),
                       task_z.detach())
        min_q = torch.min(q1, q2)

        # optimize vf
        policy_log_pi = policy_log_pi.to(tu.global_device())

        # optimize policy
        log_policy_target = min_q
        policy_loss = (policy_log_pi - log_policy_target).mean()

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean ** 2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (
                policy_log_std ** 2).mean()

        # took away pre-activation reg
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self._controller_optimizer.zero_grad()
        policy_loss.backward()
        self._controller_optimizer.step()

    def _obtain_task_samples(self,
                             runner,
                             itr,
                             num_paths):
        self._controller.reset_belief()
        total_paths = 0
        num_paths_per_batch = num_paths

        while total_paths < num_paths:
            num_samples = num_paths_per_batch * self.max_path_length
            paths = runner.obtain_samples(itr, num_samples,
                                          self._controller,
                                          self._env[self._task_idx])
            total_paths += len(paths)

            for path in paths:
                p = {
                    'states': path['states'],
                    'actions': path['actions'],
                    'env_rewards': path['env_rewards'].reshape(-1, 1),
                    'skills_onehot': path['skills_onehot'],
                    'next_states': path['next_states'],
                    'dones': path['dones'].reshape(-1, 1)
                }
                self._replay_buffers[self._task_idx].add_path(p)

    def _sample_task_path(self, indices):
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            path = self._replay_buffers[idx].sample_path()
            # TODO: trim or extend batch to the same size
            if not initialized:
                o = path['states'][np.newaxis]
                a = path['actions'][np.newaxis]
                r = path['env_rewards'][np.newaxis]
                z = path['skills_onehot'][np.newaxis]
                no = path['next_states'][np.newaxis]
                d = path['dones'][np.newaxis]
                initialized = True
            else:
                o = np.vstack((o, path['states'][np.newaxis]))
                a = np.vstack((a, path['actions'][np.newaxis]))
                r = np.vstack((r, path['env_rewards'][np.newaxis]))
                z = np.vstack((z, path['skills_onehot'][np.newaxis]))
                no = np.vstack((no, path['next_states'][np.newaxis]))
                d = np.vstack((d, path['dones'][np.newaxis]))

        o = torch.as_tensor(o, device=tu.global_device()).float()
        a = torch.as_tensor(a, device=tu.global_device()).float()
        r = torch.as_tensor(r, device=tu.global_device()).float()
        z = torch.as_tensor(z, device=tu.global_device()).float()
        no = torch.as_tensor(no, device=tu.global_device()).float()
        d = torch.as_tensor(d, device=tu.global_device()).float()

        return o, a, r, z, no, d

    def _update_target_network(self):
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

    def __getstate__(self):
        data = self.__dict__.copy()
        del data['_replay_buffers']
        return data

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._is_resuming = True

    def to(self, device=None):
        device = device or tu.global_device()
        for net in self.networks:
            # print(net)
            net.to(device)

    @classmethod
    def get_env_spec(cls, env_spec, num_skills, module):
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'controller_policy':
            in_dim = obs_dim
            out_dim = num_skills
        if module == 'qf':
            in_dim = obs_dim
            out_dim = num_skills

        in_space = akro.Box(low=-1, high=1, shape=(in_dim,), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim,),
                             dtype=np.float32)

        if module == 'controller_policy':
            spec = EnvSpec(in_space, out_space)
        if module == 'qf':
            spec = EnvSpec(in_space, out_space)
        return spec

    @property
    def policy(self):
        return self._controller

    @property
    def networks(self):
        return self._controller.networks + [self._controller] + [
            self._qf1, self._qf2, self._vf, self.target_vf]

    def get_exploration_policy(self):
        return self._controller

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        total_steps = sum(exploration_trajectories.lengths)
        o = exploration_trajectories.states
        a = exploration_trajectories.actions
        r = exploration_trajectories.env_rewards.reshape(total_steps, 1)
        s = exploration_trajectories.skills_onehot

        return self._controller


class BasicHierachWorker(DefaultWorker):
    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_number,
                 num_skills,
                 skill_actor_class,
                 controller_class):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._controller_class = controller_class
        self._skill_actor_class = skill_actor_class
        self._num_skills = num_skills
        self._skills = []
        self._states = []
        self._last_states = []
        self._cur_z = None
        self._prev_obs = None

    def start_rollout(self, skill=None):
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def step_rollout(self):
        if self._path_length < self._max_path_length:
            a, self._cur_z, agent_info = self.agent.get_action(self._prev_obs)
            self._cur_z = int(self._cur_z[0])  # get rid of [] [1]
            next_obs, r, d, env_info = self.env.step(a)
            self._states.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            self._skills.append(self._cur_z)
            for k, v in agent_info.items():
                if k == "dist":
                    continue
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(d)
            np.set_printoptions(threshold=sys.maxsize)
            if not d:
                self._prev_obs = next_obs
                return False
        self._lengths.append(self._path_length)
        self._last_states.append(self._prev_obs)
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
            if k == "dist":
                continue
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []

        # print(np.asarray(skills))
        return SkillTrajectoryBatch(env_spec=self.env.spec,
                                    num_skills=self._num_skills,
                                    skills=np.asarray(skills).reshape((np.asarray(skills).shape[0],)),
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
        while not self.step_rollout():
            pass
        return self.collect_rollout()
