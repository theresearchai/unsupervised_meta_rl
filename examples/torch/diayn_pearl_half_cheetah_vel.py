"""An example to test diayn as task proposal to pearl written in PyTorch."""

import altair as alt
import os
import click
import gym
import joblib
import numpy as np
import pandas as pd
import torch
from altair_saver import save
from torch import nn
from torch.nn import functional as F

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import GarageEnv, DiaynEnvWrapper
from garage.envs import normalize
from garage.envs.mujoco import HalfCheetahVelEnv
from garage.experiment import deterministic, LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler, SetTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.sampler import SkillWorker
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos import DIAYN
from garage.torch.algos import PEARL
from garage.torch.algos.discriminator import MLPDiscriminator
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.policies import TanhGaussianMLPSkillPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.q_functions import ContinuousMLPSkillQFunction

skills_num = 10

@wrap_experiment(snapshot_mode='gap_and_last', snapshot_gap=100)
def diayn_half_cheetah_vel_batch_for_pearl(ctxt=None, seed=1):
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(HalfCheetahVelEnv()))

    policy = TanhGaussianMLPSkillPolicy(
        env_spec=env.spec,
        skills_num=skills_num,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPSkillQFunction(env_spec=env.spec,
                                      skills_num=skills_num,
                                      hidden_sizes=[256, 256],
                                      hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPSkillQFunction(env_spec=env.spec,
                                      skills_num=skills_num,
                                      hidden_sizes=[256, 256],
                                      hidden_nonlinearity=F.relu)

    discriminator = MLPDiscriminator(env_spec=env.spec,
                                     skills_num=skills_num,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    diayn = DIAYN(env_spec=env.spec,
                  skills_num=skills_num,
                  discriminator=discriminator,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=1000,
                  max_path_length=300,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e4,
                  recorded=True,  # enable the video recording func
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=256,
                  reward_scale=1.,
                  steps_per_epoch=1)

    if torch.cuda.is_available():
        tu.set_gpu_mode(True)
    else:
        tu.set_gpu_mode(False)
    diayn.to()
    worker_args = {"skills_num": skills_num}
    runner.setup(algo=diayn, env=env, sampler_cls=LocalSkillSampler,
                 worker_class=SkillWorker, worker_args=worker_args)
    runner.train(n_epochs=1000, batch_size=1000)  # 1000
    # runner.restore(from_dir=os.path.join(os.getcwd(), 'data/local/experiment/diayn_half_cheetah_batch_50'))
    # diayn = runner.get_algo()
    runner.save(999)  # saves the last episode

    return discriminator, diayn


s = np.random.randint(0, 1000)  # 521 in the sac_cheetah example
task_proposer, diayn_trained_agent = diayn_half_cheetah_vel_batch_for_pearl(seed=s)

'''
load_dir = os.path.join(os.getcwd(), 'data/local/experiment/diayn_half_cheetah_vel_batch_for_pearl_3')
itr = 900
load_from_file = os.path.join(load_dir, 'itr_{}.pkl'.format(itr))
file = open(load_from_file, 'rb')
saved = joblib.load(file)
file.close()
diayn = saved['algo']
task_proposer = diayn.networks[1]  # _discriminator
'''
########################## hyper params for PEARL ##########################

param_num_epoches = 500
param_train_tasks_num = skills_num  # 100
param_test_tasks_num = 5 # skills_num / 2  # 30
param_encoder_hidden_size = 200
param_net_size = 300
param_num_steps_per_epoch = 2000
param_num_initial_steps = 2000
param_num_steps_prior = 400
param_num_extra_rl_steps_posterior = 600
param_batch_size = 256
param_embedding_batch_size = 100
param_embedding_mini_batch_size = 100
param_max_path_length = 300

param_latent_size = 5
param_num_tasks_sample = 5
param_meta_batch_size = 16
param_reward_scale = 5.
param_use_gpu = True


###########################################################################


@click.command()
@click.option('--num_epochs', default=param_num_epoches)
@click.option('--num_train_tasks', default=param_train_tasks_num)
@click.option('--num_test_tasks', default=param_test_tasks_num)
@click.option('--encoder_hidden_size', default=param_encoder_hidden_size)
@click.option('--net_size', default=param_net_size)
@click.option('--num_steps_per_epoch', default=param_num_steps_per_epoch)
@click.option('--num_initial_steps', default=param_num_initial_steps)
@click.option('--num_steps_prior', default=param_num_steps_prior)
@click.option('--num_extra_rl_steps_posterior',
              default=param_num_extra_rl_steps_posterior)
@click.option('--batch_size', default=param_batch_size)
@click.option('--embedding_batch_size', default=param_embedding_batch_size)
@click.option('--embedding_mini_batch_size',
              default=param_embedding_mini_batch_size)
@click.option('--max_path_length', default=param_max_path_length)
@wrap_experiment
def diayn_pearl_half_cheeth(
    ctxt=None,
    seed=1,
    num_epochs=param_num_epoches,
    num_train_tasks=param_train_tasks_num,
    num_test_tasks=param_test_tasks_num,
    latent_size=param_latent_size,
    encoder_hidden_size=param_encoder_hidden_size,
    net_size=param_net_size,
    meta_batch_size=param_meta_batch_size,
    num_steps_per_epoch=param_num_steps_per_epoch,
    num_initial_steps=param_num_initial_steps,
    num_tasks_sample=param_num_tasks_sample,
    num_steps_prior=param_num_steps_prior,
    num_extra_rl_steps_posterior=param_num_extra_rl_steps_posterior,
    batch_size=param_batch_size,
    embedding_batch_size=param_embedding_batch_size,
    embedding_mini_batch_size=param_embedding_mini_batch_size,
    max_path_length=param_max_path_length,
    reward_scale=param_reward_scale,
    use_gpu=param_use_gpu):
    if task_proposer is None:
        raise ValueError("Task proposer is empty")

    assert num_train_tasks is skills_num

    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)
    # create multi-task environment and sample tasks

    ML_train_envs = [DiaynEnvWrapper(task_proposer, skills_num, task_name,
                                     normalize(HalfCheetahVelEnv()))
                     for task_name in range(skills_num)]
    env_sampler = EnvPoolSampler(ML_train_envs)
    env = env_sampler.sample(num_train_tasks)

    # train_trajs_dist = [train_env.get_training_traj(diayn_trained_agent)
    #               for train_env in ML_train_envs]

    # ML_test_envs = [
    #     GarageEnv(normalize(
    #         DiaynEnvWrapper(env, task_proposer, skills_num, task_name)))
    #     for task_name in random.sample(range(skills_num), test_tasks_num)
    # ]

    test_env_sampler = SetTaskSampler(lambda: GarageEnv(normalize(
        HalfCheetahVelEnv())))

    runner = LocalRunner(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_path_length=max_path_length,
        reward_scale=reward_scale,
    )

    tu.set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    runner.setup(algo=pearl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=PEARLWorker)

    average_returns = runner.train(n_epochs=num_epochs, batch_size=batch_size)
    runner.save(num_epochs-1)

    return average_returns


@click.command()
@click.option('--num_epochs', default=param_num_epoches)
@click.option('--num_train_tasks', default=param_train_tasks_num)
@click.option('--num_test_tasks', default=param_test_tasks_num)
@click.option('--encoder_hidden_size', default=param_encoder_hidden_size)
@click.option('--net_size', default=param_net_size)
@click.option('--num_steps_per_epoch', default=param_num_steps_per_epoch)
@click.option('--num_initial_steps', default=param_num_initial_steps)
@click.option('--num_steps_prior', default=param_num_steps_prior)
@click.option('--num_extra_rl_steps_posterior',
              default=param_num_extra_rl_steps_posterior)
@click.option('--batch_size', default=param_batch_size)
@click.option('--embedding_batch_size',
              default=param_embedding_mini_batch_size)
@click.option('--embedding_mini_batch_size',
              default=param_embedding_mini_batch_size)
@click.option('--max_path_length', default=param_max_path_length)
@wrap_experiment
def pearl_half_cheetah(ctxt=None,
                       seed=1,
                       num_epochs=param_num_epoches,
                       num_train_tasks=param_train_tasks_num,
                       num_test_tasks=param_test_tasks_num,
                       latent_size=param_latent_size,
                       encoder_hidden_size=param_encoder_hidden_size,
                       net_size=param_net_size,
                       meta_batch_size=param_meta_batch_size,
                       num_steps_per_epoch=param_num_steps_per_epoch,
                       num_initial_steps=param_num_initial_steps,
                       num_tasks_sample=param_num_tasks_sample,
                       num_steps_prior=param_num_steps_prior,
                       num_extra_rl_steps_posterior=param_num_extra_rl_steps_posterior,
                       batch_size=param_batch_size,
                       embedding_batch_size=param_embedding_batch_size,
                       embedding_mini_batch_size=param_embedding_mini_batch_size,
                       max_path_length=param_max_path_length,
                       reward_scale=param_reward_scale,
                       use_gpu=param_use_gpu):
    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)
    # create multi-task environment and sample tasks
    env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(HalfCheetahVelEnv())))
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(lambda: GarageEnv(
        normalize(HalfCheetahVelEnv())))

    runner = LocalRunner(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_path_length=max_path_length,
        reward_scale=reward_scale,
    )

    tu.set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    runner.setup(algo=pearl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=PEARLWorker)

    average_returns = runner.train(n_epochs=num_epochs, batch_size=batch_size)
    runner.save(num_epochs - 1)

    return average_returns


def save_list_to_file(x, filename):
    with open(filename, 'w') as f:
        for item in x:
            f.write("%s\n" % item)


if not os.path.exists('tmp'):
    os.makedirs('tmp')
diayn_pearl_returns = diayn_pearl_half_cheeth()
save_list_to_file(diayn_pearl_returns, "tmp/diayn_pearl_returns.txt")
"""
pearl_returns = pearl_half_cheetah()
save_list_to_file(pearl_returns, "tmp/pearl_returns.txt")

assert (len(diayn_pearl_returns) == len(pearl_returns))

n_subject = 2
save_dir = "tmp/avg_rewards.png"
data = pd.DataFrame(
    {'Epoch': [i for i in range(len(diayn_pearl_returns) * 3)] * n_subject,
     'Subjects': ["PEARL with DIAYN" for _ in range(len(diayn_pearl_returns))]
                 + ["PEARL" for _ in range(len(pearl_returns))],
     'Average Rewards': diayn_pearl_returns + pearl_returns})
chart = alt.Chart(data).mark_line().encode(
    x='Epoch:Q',
    y='Average Rewards:Q',
    color='Subjects:N')
save(chart, save_dir)
"""
