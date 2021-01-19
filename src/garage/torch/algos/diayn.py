import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from dowel import tabular

import garage.torch.utils as tu
from garage import SkillTrajectoryBatch
from garage.misc import tensor_utils
from garage.misc.tensor_utils import discount_cumsum
from garage.torch.algos import SAC


class DIAYN(SAC):
    def __init__(self,
                 env_spec,
                 skills_num,
                 discriminator,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 *,  # Everything after this is numbers
                 max_path_length,
                 max_eval_path_length=None,
                 gradient_steps_per_itr,
                 fixed_alpha=None,  # empirically could be 0.1
                 target_entropy=None,
                 initial_log_entropy=0.,
                 discount=0.99,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 steps_per_epoch=1,
                 num_evaluation_trajectories=10,
                 eval_env=None,
                 time_per_render=90,
                 recorded=False,
                 is_gym_render=True,
                 media_save_path='diayn_tmp/',
                 media_record_epoch=100,
                 steps_skip_per_second=80):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         replay_buffer=replay_buffer,
                         max_path_length=max_path_length,
                         max_eval_path_length=max_eval_path_length,
                         gradient_steps_per_itr=gradient_steps_per_itr,
                         fixed_alpha=fixed_alpha,
                         target_entropy=target_entropy,
                         initial_log_entropy=initial_log_entropy,
                         discount=discount,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         target_update_tau=target_update_tau,
                         policy_lr=policy_lr,
                         qf_lr=qf_lr,
                         reward_scale=reward_scale,
                         optimizer=optimizer,
                         steps_per_epoch=steps_per_epoch,
                         num_evaluation_trajectories=num_evaluation_trajectories,
                         eval_env=eval_env)

        self.skills_num = skills_num
        self._prob_skills = np.full(skills_num, 1.0 / skills_num)
        self._prob_skill = 1.0 / skills_num
        self._discriminator = discriminator
        self._discriminator_optimizer = self._optimizer(
            self._discriminator.parameters(),
            lr=self._policy_lr)
        # video recording params
        self._time_per_render = time_per_render
        self._media_save_path = media_save_path
        self._recorded = recorded
        self._is_gym_render = is_gym_render
        self._video_record_epoch = media_record_epoch
        self._steps_skip_per_second = steps_skip_per_second

    def train(self, runner):
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        last_env_return = None
        last_self_return = None

        for _ in runner.step_epochs():
            for _ in range(self.steps_per_epoch):

                if not self._buffer_prefilled:
                    batch_size = int(self.min_buffer_size)
                else:
                    batch_size = None
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                path_returns = []

                for path in runner.step_path:
                    reward = self._obtain_pseudo_reward \
                        (path['states'], path['skills'])
                    self.replay_buffer.add_path(
                        dict(action=path['actions'],
                             state=path['states'],
                             next_state=path['next_states'],
                             skill=path['skills'].reshape(-1, 1),
                             skill_onehot=path['skills_onehot'],
                             env_reward=path['env_rewards'].reshape(-1, 1),
                             self_reward=reward.reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=path['dones'].reshape(-1, 1)))
                    path_returns.append(sum(reward))
                assert len(path_returns) is len(runner.step_path)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss, discriminator_loss = \
                        self._learn_once()

            last_self_return, last_env_return = self._evaluate_policy(
                runner.step_itr)
            self._log_statistics(policy_loss,
                                 qf1_loss,
                                 qf2_loss,
                                 discriminator_loss)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            if runner.step_itr % self._video_record_epoch == 0 and \
                runner.step_itr != 0:
                self.make_medias_skills(runner.step_itr)
            runner.step_itr += 1

        return np.mean(last_self_return)  # last_env_return

    def _learn_once(self, itr=None, paths=None):
        del itr
        del paths
        if self._buffer_prefilled:
            samples = self.replay_buffer.sample_transitions(
                self.buffer_batch_size)
            samples = tu.dict_np_to_torch(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(0, samples)
            self._update_targets()
            discriminator_loss = self.optimize_discriminator(samples)

        return policy_loss, qf1_loss, qf2_loss, discriminator_loss

    def optimize_policy(self, itr, samples_data):
        states = samples_data['state']
        skills = samples_data['skill_onehot']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()

        action_dists = self._policy(states, skills)
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        self._policy_optimizer.zero_grad()
        policy_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        states = samples_data['state']
        skills = samples_data['skill_onehot']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q_new_actions = torch.min(self._qf1(states, new_actions, skills),
                                      self._qf2(states, new_actions, skills))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples_data):
        states = samples_data['state']
        actions = samples_data['action']
        rewards = samples_data['self_reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_states = samples_data['next_state']
        skills = samples_data['skill_onehot']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        q1_pred = self._qf1(states, actions, skills)
        q2_pred = self._qf2(states, actions, skills)

        new_next_actions_dist = self._policy(next_states, skills)
        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        target_q_values = torch.min(
            self._target_qf1(next_states, new_next_actions, skills),
            self._target_qf2(
                next_states, new_next_actions, skills)).flatten() - (
                              alpha * new_log_pi)

        with torch.no_grad():
            q_target = rewards * self.reward_scale + (
                1. - terminals) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    def _discriminator_objective(self, samples_data):
        states = samples_data['next_state']

        discriminator_pred = self._discriminator(states)
        discriminator_target = samples_data['skill_onehot']
        # print(discriminator_pred.shape)
        # print(discriminator_target.shape)

        discriminator_loss = F.mse_loss(discriminator_pred.flatten(),
                                        discriminator_target.flatten())

        return discriminator_loss

    def optimize_discriminator(self, samples_data):
        discriminator_loss = self._discriminator_objective(samples_data)

        self._discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self._discriminator_optimizer.step()

        return discriminator_loss

    def _evaluate_policy(self, epoch):
        # TODO: picks the most often policy and runs it - enable OpenGL visualization (saves as clip)
        eval_trajectories = self._obtain_evaluation_samples(
            self._eval_env,
            num_trajs=self._num_evaluation_trajectories)
        last_return = self._log_performance(itr=epoch,
                                            batch=eval_trajectories,
                                            discount=self.discount)
        return last_return

    def make_medias_skills(self, current_epoch):
        import os
        if not os.path.exists(self._media_save_path):
            os.makedirs(self._media_save_path)

        max_path_length = self.max_eval_path_length
        if max_path_length is None or np.isinf(max_path_length):
            max_path_length = 1000
        for skill in range(self.skills_num):
            if self._is_gym_render is True:
                filename = "epoch{}_skill{}.mp4".format(current_epoch, skill)
            else:
                filename = "epoch{}_skill{}.png".format(current_epoch, skill)

            _ = self._rollout(self._eval_env,
                              self.policy,
                              skill=skill,
                              max_path_length=max_path_length,
                              deterministic=True,
                              recorded=True,
                              save_media_filename="{}{}".format(
                                  self._media_save_path,
                                  filename))

    def _obtain_evaluation_samples(self, env, num_trajs=100):
        paths = []
        max_path_length = self.max_eval_path_length
        if max_path_length is None:
            max_path_length = self.max_path_length
        # Use a finite length rollout for evaluation.
        if max_path_length is None or np.isinf(max_path_length):
            max_path_length = 1000

        for _ in range(num_trajs):
            path = self._rollout(env,
                                 self.policy,
                                 max_path_length=max_path_length,
                                 deterministic=True)
            paths.append(path)

        return SkillTrajectoryBatch.from_trajectory_list(self.env_spec,
                                                         self.skills_num,
                                                         paths)

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss,
                        discriminator_loss):
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Discriminator/Loss', float(discriminator_loss))
        tabular.record('Policy/Loss', policy_loss.item())
        tabular.record('QF/{}'.format('Qf1Loss'), float(qf1_loss))
        tabular.record('QF/{}'.format('Qf2Loss'), float(qf2_loss))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    @property
    def networks(self):
        return [
            self._policy, self._discriminator, self._qf1, self._qf2,
            self._target_qf1, self._target_qf2
        ]

    def _sample_skill(self):  # uniform dist. in order to maximize entropy
        return np.random.choice(self.skills_num, p=self._prob_skills)

    def _obtain_pseudo_reward(self, states, skills):
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        if isinstance(skills, int):
            skills = np.array([skills])

        q = self._discriminator(states).detach().cpu().numpy()
        q_z = np.array([q[i, skills[i]] for i in range(skills.shape[0])])
        reward = np.log(q_z) - np.log(np.full(q_z.shape, self._prob_skill))

        return reward

    def _rollout(self,
                 env,
                 agent,  # self.policy
                 *,
                 skill=-1,
                 max_path_length=np.inf,
                 animated=False,
                 recorded=False,
                 save_media_filename=None,
                 speedup=1,
                 deterministic=False):

        # TODO: sanity check if agent isinstance of skill algo

        if skill is -1:
            skill = self._sample_skill()
        skills = []
        states = []
        actions = []
        self_rewards = []
        env_rewards = []
        agent_infos = []
        env_infos = []
        dones = []

        s = env.reset()
        z = np.eye(self.skills_num)[skill]
        agent.reset()
        path_length = 0

        video_buffer = []
        if recorded and self._is_gym_render is True:
            video_buffer.append(env.render(mode="rgb_array"))

        if animated and self._is_gym_render is True:
            env.render(mode="human")

        while path_length < (max_path_length or np.inf):
            s = env.observation_space.flatten(s)
            a, agent_info = agent.get_action(s, z)
            if deterministic and 'mean' in agent_infos:
                a = agent_info['mean']
            next_s, env_r, d, env_info = env.step(a)
            self_r = self._obtain_pseudo_reward(s, skill)
            states.append(s)
            self_rewards.append(self_r)
            env_rewards.append(env_r)
            actions.append(a)
            skills.append(skill)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            dones.append(d)
            path_length += 1
            if d:
                break
            s = next_s
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
            if recorded and self._is_gym_render is True:
                video_buffer.append(env.render(mode="rgb_array"))

        if recorded and self._is_gym_render is False:
            env.render(
                paths=[dict(
                    skills=skills,
                    states=states,
                    actions=actions,
                    self_rewards=self_rewards,  # squeeze column
                    env_rewards=env_rewards,
                    agent_infos=agent_infos,
                    env_infos=env_infos,
                    dones=dones)],
                save_path=save_media_filename
            )

        if recorded and self._is_gym_render is True:
            fps = (1 / self._time_per_render)
            self._save_video(video_buffer, save_media_filename, fps)

        return dict(
            skills=skills,
            states=np.array(states),
            actions=np.array(actions),
            self_rewards=np.array(self_rewards)[:, 0],  # squeeze column
            env_rewards=np.array(env_rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            dones=np.array(dones),
        )

    def _save_video(self, rgb_array, filename, fps):
        writer = imageio.get_writer(filename, fps=fps)
        for idx in range(0, len(rgb_array), self._steps_skip_per_second):
            writer.append_data(rgb_array[idx])
        writer.close()

    def _log_performance(self, itr, batch, discount, prefix='Evaluation'):
        self_returns = []
        env_returns = []
        undiscounted_self_returns = []
        undiscounted_env_returns = []
        completion = []
        success = []
        for trajectory in batch.split():
            self_returns.append(
                discount_cumsum(trajectory.self_rewards, discount))
            env_returns.append(
                discount_cumsum(trajectory.env_rewards, discount))
            undiscounted_self_returns.append(sum(trajectory.self_rewards))
            undiscounted_env_returns.append(sum(trajectory.env_rewards))
            completion.append(float(trajectory.terminals.any()))
            if 'success' in trajectory.env_infos:
                success.append(float(trajectory.env_infos['success'].any()))

        average_discounted_self_return = np.mean(
            [rtn[0] for rtn in self_returns])
        average_discounted_env_return = np.mean(
            [rtn[0] for rtn in env_returns])

        with tabular.prefix(prefix + '/'):
            tabular.record('Iteration', itr)
            tabular.record('NumTrajs', len(self_returns))
            # pseudo reward
            tabular.record('AverageDiscountedSelfReturn',
                           average_discounted_self_return)
            tabular.record('AverageSelfReturn',
                           np.mean(undiscounted_self_returns))
            tabular.record('StdSelfReturn', np.std(undiscounted_self_returns))
            tabular.record('MaxSelfReturn', np.max(undiscounted_self_returns))
            tabular.record('MinSelfReturn', np.min(undiscounted_self_returns))
            # env reward
            tabular.record('AverageDiscountedEnvReturn',
                           average_discounted_env_return)
            tabular.record('AverageEnvReturn',
                           np.mean(undiscounted_env_returns))
            tabular.record('StdEnvReturn', np.std(undiscounted_env_returns))
            tabular.record('MaxEnvReturn', np.max(undiscounted_env_returns))
            tabular.record('MinEnvReturn', np.min(undiscounted_env_returns))

            tabular.record('CompletionRate', np.mean(completion))
            if success:
                tabular.record('SuccessRate', np.mean(success))

        return undiscounted_self_returns, undiscounted_env_returns
