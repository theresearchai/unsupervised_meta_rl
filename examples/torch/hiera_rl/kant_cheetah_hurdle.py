import faulthandler
import os

from garage.envs.hierarchical_rl_gym import HalfCheetahEnv_Hurdle
from garage.torch.algos.kant import Kant

faulthandler.enable()

import click
import joblib
import numpy as np
from torch.nn import functional

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import DiaynEnvWrapper
from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos.meta_kant import KantWorker
from garage.torch.modules.categorical_mlp import CategoricalMLPPolicy
from garage.torch.policies.context_conditioned_controller_policy import \
    OpenContextConditionedControllerPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

seed = np.random.randint(0, 1000)
skills_num = 10

load_dir = os.path.join(os.getcwd(),
                        'data/local/experiment/diayn_cheetah_hurdle_27')
itr = 900
load_from_file = os.path.join(load_dir, 'itr_{}.pkl'.format(itr))
file = open(load_from_file, 'rb')
saved = joblib.load(file)
file.close()
skill_env = saved['env']
diayn = saved['algo']
skill_actor = diayn.networks[0]  # _policy
task_proposer = diayn.networks[1]  # _discriminator

param_num_epoches = 500
param_train_tasks_num = skills_num  # 100
param_test_tasks_num = 5  # skills_num / 2  # 30
param_encoder_hidden_size = 200
param_net_size = 300
param_num_steps_per_epoch = 300
param_num_initial_steps = 300
param_num_skills_reason_steps = 300
param_num_steps_prior = 300
param_num_extra_rl_steps_posterior = 300
param_num_skills_sample = 10
param_batch_size = 256
param_embedding_batch_size = 100
param_embedding_mini_batch_size = 100
param_max_path_length = 300
param_latent_size = 5
param_num_tasks_sample = 5
param_meta_batch_size = 16
param_skills_reason_reward_scale = 1
param_tasks_adapt_reward_scale = 1.2
param_use_gpu = True


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
@click.option('--num_skills_sample', default=param_num_skills_sample)
@click.option('--num_skills_reason_steps',
              default=param_num_skills_reason_steps)
@click.option('--batch_size', default=param_batch_size)
@click.option('--embedding_batch_size', default=param_embedding_batch_size)
@click.option('--embedding_mini_batch_size',
              default=param_embedding_mini_batch_size)
@click.option('--max_path_length', default=param_max_path_length)
@wrap_experiment(snapshot_mode='gap_and_last', snapshot_gap=100)
def kant_cheetah_hurdle(ctxt=None,
                          seed=seed,
                          num_skills=skills_num,
                          num_epochs=param_num_epoches,
                          num_train_tasks=param_train_tasks_num,
                          num_test_tasks=param_test_tasks_num,
                          is_encoder_recurrent=False,
                          latent_size=param_latent_size,
                          encoder_hidden_size=param_encoder_hidden_size,
                          net_size=param_net_size,
                          meta_batch_size=param_meta_batch_size,
                          num_steps_per_epoch=param_num_steps_per_epoch,
                          num_initial_steps=param_num_initial_steps,
                          num_tasks_sample=param_num_tasks_sample,
                          num_steps_prior=param_num_steps_prior,
                          num_extra_rl_steps_posterior=param_num_extra_rl_steps_posterior,
                          num_skills_sample=param_num_skills_sample,
                          num_skills_reason_steps=param_num_skills_reason_steps,
                          batch_size=param_batch_size,
                          embedding_batch_size=param_embedding_batch_size,
                          embedding_mini_batch_size=param_embedding_mini_batch_size,
                          max_path_length=param_max_path_length,
                          skills_reason_reward_scale=param_skills_reason_reward_scale,
                          tasks_adapt_reward_scale=param_tasks_adapt_reward_scale,
                          use_gpu=param_use_gpu):
    assert num_train_tasks is skills_num

    set_seed(seed)

    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)

    ML_train_envs = [DiaynEnvWrapper(task_proposer, skills_num, task_name,
                                     normalize(HalfCheetahEnv_Hurdle()))
                     for task_name in range(skills_num)]

    env_sampler = EnvPoolSampler(ML_train_envs)
    env = env_sampler.sample(num_train_tasks)

    runner = LocalRunner(ctxt)

    qf_env = Kant.get_env_spec(env[0](), latent_size, num_skills, "qf")

    qf = ContinuousMLPQFunction(env_spec=qf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = Kant.get_env_spec(env[0](), latent_size, num_skills, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    controller_policy_env = Kant.get_env_spec(env[0](), latent_size,
                                              module="controller_policy",
                                              num_skills=num_skills)

    controller_policy = CategoricalMLPPolicy(env_spec=controller_policy_env,
                                             hidden_sizes=[net_size, net_size],
                                             hidden_nonlinearity=functional.relu)

    kant = Kant(
        env=env,
        skill_env=skill_env,
        controller_policy=controller_policy,
        skill_actor=skill_actor,
        qf=qf,
        vf=vf,
        num_skills=num_skills,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        is_encoder_recurrent=is_encoder_recurrent,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        meta_batch_size=meta_batch_size,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_per_epoch=num_steps_per_epoch,
        num_steps_prior=num_steps_prior,  # num_steps_posterior
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        num_skills_reason_steps=num_skills_reason_steps,
        num_skills_sample=num_skills_sample,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_path_length=max_path_length,
        skills_reason_reward_scale=skills_reason_reward_scale,
        tasks_adapt_reward_scale=tasks_adapt_reward_scale
    )

    tu.set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        kant.to()

    worker_args = dict(num_skills=num_skills,
                       skill_actor_class=type(skill_actor),
                       controller_class=OpenContextConditionedControllerPolicy,
                       deterministic=False, accum_context=True)

    runner.setup(algo=kant,
                 env=env[0](),
                 sampler_cls=LocalSkillSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=KantWorker,
                 worker_args=worker_args
                 )

    average_returns = runner.train(n_epochs=num_epochs, batch_size=batch_size)
    runner.save(num_epochs - 1)

    return average_returns


kant_returns = kant_cheetah_hurdle()


def save_list_to_file(x, filename):
    with open(filename, 'w') as f:
        for item in x:
            f.write("%s\n" % item)


if not os.path.exists('tmp'):
    os.makedirs('tmp')
save_list_to_file(kant_returns, "tmp/kant_cheetah_hurdle_returns.txt")
