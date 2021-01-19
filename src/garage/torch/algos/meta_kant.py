import copy
import sys
from collections import defaultdict

import akro
from dowel import logger
import torch
import torch.nn.functional as F
import numpy as np

from garage import TimeStep, SkillTrajectoryBatch, InOutSpec, SkillTimeStep
from garage.envs import EnvSpec
from garage.experiment import MetaEvaluator
from garage.np.algos import MetaRLAlgorithm
from garage.replay_buffer import PathBuffer
from garage.sampler import DefaultWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies.context_conditioned_controller_policy import \
    OpenContextConditionedControllerPolicy, GaussianContextEncoder
import garage.torch.utils as tu


class MetaKant(MetaRLAlgorithm):
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
                 latent_dim,
                 encoder_hidden_sizes,
                 test_env_sampler,
                 sampler_class,  # to avoid cycling import
                 controller_class=OpenContextConditionedControllerPolicy,
                 encoder_class=GaussianContextEncoder,
                 encoder_module_class=MLPEncoder,
                 is_encoder_recurrent=False,
                 controller_lr=3E-4,
                 qf_lr=3E-4,
                 vf_lr=3E-4,
                 context_lr=3E-4,
                 policy_mean_reg_coeff=1E-3,
                 policy_std_reg_coeff=1E-3,
                 policy_pre_activation_coeff=0.,
                 soft_target_tau=0.005,
                 kl_lambda=.1,
                 optimizer_class=torch.optim.Adam,
                 use_next_obs_in_context=False,
                 meta_batch_size=64,
                 num_steps_per_epoch=1000,
                 num_skills_reason_steps=1000,
                 num_skills_sample=10,
                 num_initial_steps=1500,
                 num_tasks_sample=5,
                 num_steps_prior=400,
                 num_steps_posterior=0,
                 num_extra_rl_steps_posterior=600,
                 batch_size=1024,
                 embedding_batch_size=1024,
                 embedding_mini_batch_size=1024,
                 max_path_length=1000,
                 discount=0.99,
                 replay_buffer_size=1000000,
                 # TODO: the ratio needs to be tuned
                 skills_reason_reward_scale=1,
                 tasks_adapt_reward_scale=1.2,
                 use_information_bottleneck=True,
                 update_post_train=1,
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
        self._latent_dim = latent_dim

        self._policy_mean_reg_coeff = policy_mean_reg_coeff
        self._policy_std_reg_coeff = policy_std_reg_coeff
        self._policy_pre_activation_coeff = policy_pre_activation_coeff
        self._soft_target_tau = soft_target_tau
        self._kl_lambda = kl_lambda
        self._use_next_obs_in_context = use_next_obs_in_context

        self._meta_batch_size = meta_batch_size
        self._num_skills_reason_steps = num_skills_reason_steps
        self._num_steps_per_epoch = num_steps_per_epoch
        self._num_initial_steps = num_initial_steps
        self._num_tasks_sample = num_tasks_sample
        self._num_skills_sample = num_skills_sample
        self._num_steps_prior = num_steps_prior
        self._num_steps_posterior = num_steps_posterior
        self._num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self._batch_size = batch_size
        self._embedding_batch_size = embedding_batch_size
        self._embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self._discount = discount
        self._replay_buffer_size = replay_buffer_size

        self._is_encoder_recurrent = is_encoder_recurrent
        self._use_information_bottleneck = use_information_bottleneck
        self._skills_reason_reward_scale = skills_reason_reward_scale
        self._tasks_adapt_reward_scale = tasks_adapt_reward_scale
        self._update_post_train = update_post_train
        self._task_idx = None
        self._skill_idx = None  # do we really need it

        self._is_resuming = False

        worker_args = dict(num_skills=num_skills,
                           skill_actor_class=type(skill_actor),
                           controller_class=controller_class,
                           deterministic=False, accum_context=True)
        self._evaluator = MetaEvaluator(test_task_sampler=test_env_sampler,
                                        max_path_length=max_path_length,
                                        worker_class=KantWorker,
                                        worker_args=worker_args,
                                        n_test_tasks=num_test_tasks,
                                        sampler_class=sampler_class,
                                        trajectory_batch_class=SkillTrajectoryBatch)
        self._average_rewards = []

        encoder_spec = self.get_env_spec(env[0](), latent_dim, num_skills,
                                         'encoder')
        encoder_in_dim = int(np.prod(encoder_spec.input_space.shape))
        encoder_out_dim = int(np.prod(encoder_spec.output_space.shape))

        encoder_module = encoder_module_class(input_dim=encoder_in_dim,
                                              output_dim=encoder_out_dim,
                                              hidden_sizes=encoder_hidden_sizes)

        context_encoder = encoder_class(encoder_module,
                                        use_information_bottleneck,
                                        latent_dim)

        self._controller = controller_class(
            latent_dim=latent_dim,
            context_encoder=context_encoder,
            controller_policy=controller_policy,
            num_skills=num_skills,
            sub_actor=skill_actor,
            use_information_bottleneck=use_information_bottleneck,
            use_next_obs=use_next_obs_in_context)

        self._skills_replay_buffer = PathBuffer(replay_buffer_size)

        self._replay_buffers = {
            i: PathBuffer(replay_buffer_size)
            for i in range(num_train_tasks)
        }

        self._context_replay_buffers = {
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
        self.context_optimizer = optimizer_class(
            self._controller.networks[0].parameters(),
            lr=context_lr,
        )

    def train(self, runner):
        for _ in runner.step_epochs():
            epoch = runner.step_itr / self._num_steps_per_epoch

            if epoch == 0 or self._is_resuming:
                for idx in range(self._num_skills):
                    self._skill_idx = idx
                    self._obtain_skill_samples(runner, epoch,
                                               self._num_initial_steps)
                for idx in range(self._num_train_tasks):
                    self._task_idx = idx
                    self._obtain_task_samples(runner, epoch,
                                              self._num_initial_steps, np.inf)
                self._is_resuming = False

            logger.log('Sampling skills')
            for idx in range(self._num_skills_sample):
                self._skill_idx = idx
                # self._skills_replay_buffer.clear()
                self._obtain_skill_samples(runner, epoch,
                                           self._num_skills_reason_steps)

            logger.log('Training skill reasoning...')
            self._skills_reason_train_once()

            logger.log('Sampling tasks')
            for _ in range(self._num_tasks_sample):
                idx = np.random.randint(self._num_train_tasks)
                self._task_idx = idx
                self._context_replay_buffers[idx].clear()
                # obtain samples with z ~ prior
                logger.log("Obtaining samples with z ~ prior")
                if self._num_steps_prior > 0:
                    self._obtain_task_samples(runner, epoch, self._num_steps_prior,
                                              np.inf)
                # obtain samples with z ~ posterior
                logger.log("Obtaining samples with z ~ posterior")
                if self._num_steps_posterior > 0:
                    self._obtain_task_samples(runner, epoch,
                                              self._num_steps_posterior,
                                              self._update_post_train)
                # obtain extras samples for RL training but not encoder
                logger.log("Obtaining extra samples for RL traing but not encoder")
                if self._num_extra_rl_steps_posterior > 0:
                    self._obtain_task_samples(runner,
                                              epoch,
                                              self._num_extra_rl_steps_posterior,
                                              self._update_post_train,
                                              add_to_enc_buffer=False)

            logger.log('Training task adapting...')
            self._tasks_adapt_train_once()

            runner.step_itr += 1

            logger.log('Evaluating...')
            # evaluate
            self._controller.reset_belief()
            self._average_rewards.append(self._evaluator.evaluate(self))

        return self._average_rewards

    def _skills_reason_train_once(self):
        for _ in range(self._num_steps_per_epoch):
            self._skills_reason_optimize_policy()

    def _tasks_adapt_train_once(self):
        for _ in range(self._num_steps_per_epoch):
            indices = np.random.choice(range(self._num_train_tasks),
                                       self._meta_batch_size)
            self._tasks_adapt_optimize_policy(indices)

    def _skills_reason_optimize_policy(self):
        self._controller.reset_belief()

        # data shape is (task, batch, feat)
        obs, actions, rewards, skills, next_obs, terms, context = self.\
            _sample_skill_path()

        # skills_pred is distribution
        policy_outputs, skills_pred, task_z = self._controller(obs, context)

        _, policy_mean, policy_log_std, policy_log_pi = policy_outputs[:4]

        self.context_optimizer.zero_grad()
        if self._use_information_bottleneck:
            kl_div = self._controller.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        skills_target = skills.clone().detach().requires_grad_(True)\
            .to(tu.global_device())
        skills_pred = skills_pred.to(tu.global_device())

        policy_loss = F.mse_loss(skills_pred.flatten(), skills_target.flatten())\
                      * self._skills_reason_reward_scale

        mean_reg_loss = self._policy_mean_reg_coeff * (policy_mean ** 2).mean()
        std_reg_loss = self._policy_std_reg_coeff * (
                policy_log_std ** 2).mean()

        #took away the pre-activation reg term
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self._controller_optimizer.zero_grad()
        policy_loss.backward()
        self._controller_optimizer.step()

    def _tasks_adapt_optimize_policy(self, indices):
        num_tasks = len(indices)
        obs, actions, rewards, skills, next_obs, terms, context = \
            self._sample_task_path(indices)
        self._controller.reset_belief(num_tasks=num_tasks)

        # data shape is (task, batch, feat)
        # new_skills_pred is distribution
        policy_outputs, new_skills_pred, task_z = self._controller(obs, context)
        new_actions, policy_mean, policy_log_std, policy_log_pi = policy_outputs[:4]

        # flatten out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        skills = skills.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize qf and encoder networks
        # TODO try [obs, skills, actions] or [obs, skills, task_z]
        # FIXME prob need to reshape or tile task_z
        obs = obs.to(tu.global_device())
        skills = skills.to(tu.global_device())
        next_obs = next_obs.to(tu.global_device())

        q1_pred = self._qf1(torch.cat([obs, skills], dim=1), task_z)
        q2_pred = self._qf2(torch.cat([obs, skills], dim=1), task_z)
        v_pred = self._vf(obs, task_z.detach())

        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self._use_information_bottleneck:
            kl_div = self._controller.compute_kl_div()
            kl_loss = self._kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()

        rewards_flat = rewards.view(b * t, -1)
        rewards_flat = rewards_flat * self._tasks_adapt_reward_scale
        terms_flat = terms.view(b * t, -1)
        q_target = rewards_flat + (
            1. - terms_flat) * self._discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean(
            (q2_pred - q_target) ** 2)
        qf_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        new_skills_pred = new_skills_pred.to(tu.global_device())

        # compute min Q on the new actions
        q1 = self._qf1(torch.cat([obs, new_skills_pred], dim=1),
                       task_z.detach())
        q2 = self._qf2(torch.cat([obs, new_skills_pred], dim=1),
                       task_z.detach())
        min_q = torch.min(q1, q2)

        # optimize vf
        policy_log_pi = policy_log_pi.to(tu.global_device())

        v_target = min_q - policy_log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

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

    def _obtain_skill_samples(self,
                              runner,
                              itr,
                              num_paths):
        self._controller.reset_belief()
        total_paths = 0

        while total_paths < num_paths:
            num_samples = num_paths * self.max_path_length
            paths = runner.obtain_samples(itr, num_samples,
                                          self._skill_actor,
                                          self._skill_env)
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
                self._skills_replay_buffer.add_path(p)

    def _obtain_task_samples(self,
                             runner,
                             itr,
                             num_paths,
                             update_posterior_rate,
                             add_to_enc_buffer=True):
        self._controller.reset_belief()
        total_paths = 0

        if update_posterior_rate != np.inf:
            num_paths_per_batch = update_posterior_rate
        else:
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

                if add_to_enc_buffer:
                    self._context_replay_buffers[self._task_idx].add_path(p)

            if update_posterior_rate != np.inf:
                context = self._sample_path_context(self._task_idx)
                self._controller.infer_posterior(context)

    def _sample_task_path(self, indices):
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            path = self._context_replay_buffers[idx].sample_path()
            # should be replay_buffers[]
            # TODO: trim or extend batch to the same size

            context_o = path['states']
            context_a = path['actions']
            context_r = path['env_rewards']
            context_z = path['skills_onehot']
            context = np.hstack((np.hstack((np.hstack((context_o, context_a)),
                                            context_r)), context_z))
            if self._use_next_obs_in_context:
                context = np.hstack((context, path['next_states']))

            if not initialized:
                final_context = context[np.newaxis]
                o = path['states'][np.newaxis]
                a = path['actions'][np.newaxis]
                r = path['env_rewards'][np.newaxis]
                z = path['skills_onehot'][np.newaxis]
                no = path['next_states'][np.newaxis]
                d = path['dones'][np.newaxis]
                initialized = True
            else:
                # print(o.shape)
                # print(path['states'].shape)
                o = np.vstack((o, path['states'][np.newaxis]))
                a = np.vstack((a, path['actions'][np.newaxis]))
                r = np.vstack((r, path['env_rewards'][np.newaxis]))
                z = np.vstack((z, path['skills_onehot'][np.newaxis]))
                no = np.vstack((no, path['next_states'][np.newaxis]))
                d = np.vstack((d, path['dones'][np.newaxis]))
                final_context = np.vstack((final_context, context[np.newaxis]))

        o = torch.as_tensor(o, device=tu.global_device()).float()
        a = torch.as_tensor(a, device=tu.global_device()).float()
        r = torch.as_tensor(r, device=tu.global_device()).float()
        z = torch.as_tensor(z, device=tu.global_device()).float()
        no = torch.as_tensor(no, device=tu.global_device()).float()
        d = torch.as_tensor(d, device=tu.global_device()).float()
        final_context = torch.as_tensor(final_context,
                                        device=tu.global_device()).float()
        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return o, a, r, z, no, d, final_context

    def _sample_skill_path(self):
        path = self._skills_replay_buffer.sample_path()
        # TODO: trim or extend batch to the same size
        o = path['states']
        a = path['actions']
        r = path['env_rewards']
        z = path['skills_onehot']
        context = np.hstack((np.hstack((np.hstack((o, a)), r)), z))
        if self._use_next_obs_in_context:
            context = np.hstack((context, path['next_states']))

        context = context[np.newaxis]
        o = path['states'][np.newaxis]
        a = path['actions'][np.newaxis]
        r = path['env_rewards'][np.newaxis]
        z = path['skills_onehot'][np.newaxis]
        no = path['next_states'][np.newaxis]
        d = path['dones'][np.newaxis]

        o = torch.as_tensor(o, device=tu.global_device()).float()
        a = torch.as_tensor(a, device=tu.global_device()).float()
        r = torch.as_tensor(r, device=tu.global_device()).float()
        z = torch.as_tensor(z, device=tu.global_device()).float()
        no = torch.as_tensor(no, device=tu.global_device()).float()
        d = torch.as_tensor(d, device=tu.global_device()).float()
        context = torch.as_tensor(context, device=tu.global_device()).float()
        context = context.unsqueeze(0)

        return o, a, r, z, no, d, context

    def _sample_path_context(self, indices):
        if not hasattr(indices, '__iter__'):
            indices = [indices]

        initialized = False
        for idx in indices:
            path = self._context_replay_buffers[idx].sample_path()
            o = path['states']
            a = path['actions']
            r = path['env_rewards']
            z = path['skills_onehot']
            context = np.hstack((np.hstack((np.hstack((o, a)), r)), z))
            if self._use_next_obs_in_context:
                context = np.hstack((context, path['states']))

            if not initialized:
                final_context = context[np.newaxis]
                initialized = True
            else:
                final_context = np.vstack((final_context, context[np.newaxis]))

        final_context = torch.as_tensor(final_context,
                                        device=tu.global_device()).float()
        if len(indices) == 1:
            final_context = final_context.unsqueeze(0)

        return final_context

    def _update_target_network(self):
        for target_param, param in zip(self.target_vf.parameters(),
                                       self._vf.parameters()):
            target_param.data.copy_(target_param.data *
                                    (1.0 - self._soft_target_tau) +
                                    param.data * self._soft_target_tau)

    def __getstate__(self):
        data = self.__dict__.copy()
        del data['_skills_replay_buffer']
        del data['_replay_buffers']
        del data['_context_replay_buffers']
        return data

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._skills_replay_buffer = PathBuffer(self._replay_buffer_size)

        self._replay_buffers = {
            i: PathBuffer(self._replay_buffer_size)
            for i in range(self._num_train_tasks)
        }

        self._context_replay_buffers = {
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
    def get_env_spec(cls, env_spec, latent_dim, num_skills, module):
        obs_dim = int(np.prod(env_spec.observation_space.shape))
        # print("obs_dim is")
        # print(obs_dim)
        action_dim = int(np.prod(env_spec.action_space.shape))
        if module == 'encoder':
            in_dim = obs_dim + action_dim + num_skills + 1
            out_dim = latent_dim * 2
        elif module == 'vf':
            in_dim = obs_dim
            out_dim = latent_dim
        elif module == 'controller_policy':
            in_dim = obs_dim + latent_dim
            out_dim = num_skills
        elif module == 'qf':
            in_dim = obs_dim + latent_dim
            out_dim = num_skills

        in_space = akro.Box(low=-1, high=1, shape=(in_dim,), dtype=np.float32)
        out_space = akro.Box(low=-1,
                             high=1,
                             shape=(out_dim,),
                             dtype=np.float32)

        if module == 'encoder':
            spec = InOutSpec(in_space, out_space)
        elif module == 'vf':
            spec = EnvSpec(in_space, out_space)
        elif module == 'controller_policy':
            spec = EnvSpec(in_space, out_space)
        elif module == 'qf':
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
        ctxt = np.hstack((o, a, r, s)).reshape(1, total_steps, -1)
        context = torch.as_tensor(ctxt, device=tu.global_device()).float()
        self._controller.infer_posterior(context)

        return self._controller


class KantWorker(DefaultWorker):
    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_number,
                 num_skills,
                 skill_actor_class,
                 controller_class,
                 deterministic=False,
                 accum_context=True):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._deterministic = deterministic
        self._accum_context = accum_context
        self._controller_class = controller_class
        self._skill_actor_class = skill_actor_class
        self._num_skills = num_skills
        self._skills = []
        self._states = []
        self._last_states = []
        self._cur_z = None
        self._prev_obs = None

    def start_rollout(self, skill=None):
        # print("agent")
        # print(type(self.agent))
        # print("controller_class")
        # print(self._controller_class)
        # print("skill_actor")
        # print(self._skill_actor_class)
        if isinstance(self.agent, self._skill_actor_class):
            if skill is None:
                prob_skill = np.full(self._num_skills, 1.0 / self._num_skills)
                self._cur_z = np.random.choice(self._num_skills, p=prob_skill)
            else:
                self._cur_z = skill
        elif isinstance(self.agent, self._controller_class):
            pass
        else:
            raise ValueError("Agent in KantWorker has been updated to an"
                             "unknown class")
        self._path_length = 0
        self._prev_obs = self.env.reset()

    def step_rollout(self):
        if self._path_length < self._max_path_length:
            if isinstance(self.agent, self._skill_actor_class):
                z_onehot = np.eye(self._num_skills)[self._cur_z]
                a, agent_info = self.agent.get_action(self._prev_obs, z_onehot)
            elif isinstance(self.agent, self._controller_class):
                a, self._cur_z, agent_info = self.agent.get_action(self._prev_obs)
                self._cur_z = int(self._cur_z[0])  # get rid of [] [1]
                if self._deterministic:  # not supported
                    a = agent_info['mean']
            else:
                raise ValueError("Agent in KantWorker has been updated to an"
                                 "unknown class")
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
            if isinstance(self.agent, self._controller_class) and self._accum_context:
                s = SkillTimeStep(env_spec=self.env,
                                  state=self._prev_obs,
                                  next_state=next_obs,
                                  skill=self._cur_z,
                                  num_skills=self._num_skills,
                                  action=a,
                                  reward=float(r),
                                  terminal=d,
                                  env_info=env_info,
                                  agent_info=agent_info)
                self.agent.update_context(s)
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
        if isinstance(self.agent, self._controller_class):
            self.agent.sample_from_belief()
        self.start_rollout(skill)
        while not self.step_rollout():
            pass
        if isinstance(self.agent, self._controller_class):
            self._agent_infos['context'] = [self.agent.z.detach().cpu().numpy()
                                            ] * sum(self._lengths)
        return self.collect_rollout()
