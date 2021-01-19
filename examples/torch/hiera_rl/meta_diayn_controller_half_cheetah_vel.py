import faulthandler;
import os

from garage.torch.algos.basic_hierach import MetaBasicHierch, \
    BasicHierachWorker
from garage.torch.policies.controller_policy import ControllerPolicy

faulthandler.enable()

import click
import joblib
import numpy as np
from torch.nn import functional

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import GarageEnv, DiaynEnvWrapper
from garage.envs import normalize
from garage.envs.mujoco import HalfCheetahVelEnv
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler, SetTaskSampler
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.modules.categorical_mlp import CategoricalMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction

seed = np.random.randint(0, 1000)
skills_num = 10

load_dir = os.path.join(os.getcwd(),
                        'data/local/experiment/diayn_half_cheetah_vel_batch_for_pearl_3')
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
param_net_size = 300
param_num_steps_per_epoch = 1000
param_num_skills_sample = 10
param_batch_size = 256
param_embedding_batch_size = 100
param_embedding_mini_batch_size = 100
param_max_path_length = 300
param_latent_size = 5
param_num_tasks_sample = 5
param_meta_batch_size = 16
param_use_gpu = True


@click.command()
@click.option('--num_epochs', default=param_num_epoches)
@click.option('--num_train_tasks', default=param_train_tasks_num)
@click.option('--num_test_tasks', default=param_test_tasks_num)
@click.option('--net_size', default=param_net_size)
@click.option('--num_steps_per_epoch', default=param_num_steps_per_epoch)
@click.option('--num_skills_sample', default=param_num_skills_sample)
@click.option('--batch_size', default=param_batch_size)
@click.option('--embedding_batch_size', default=param_embedding_batch_size)
@click.option('--embedding_mini_batch_size',
              default=param_embedding_mini_batch_size)
@click.option('--max_path_length', default=param_max_path_length)
@wrap_experiment(snapshot_mode='gap and last', snapshot_gap=100)
def meta_basic_hierachy_cheetah_vel(ctxt=None,
                                    seed=seed,
                                    num_skills=skills_num,
                                    num_epochs=param_num_epoches,
                                    num_train_tasks=param_train_tasks_num,
                                    num_test_tasks=param_test_tasks_num,
                                    net_size=param_net_size,
                                    meta_batch_size=param_meta_batch_size,
                                    num_steps_per_epoch=param_num_steps_per_epoch,
                                    num_tasks_sample=param_num_tasks_sample,
                                    batch_size=param_batch_size,
                                    embedding_batch_size=param_embedding_batch_size,
                                    embedding_mini_batch_size=param_embedding_mini_batch_size,
                                    max_path_length=param_max_path_length,
                                    use_gpu=param_use_gpu):
    assert num_train_tasks is skills_num

    set_seed(seed)

    ML_train_envs = [DiaynEnvWrapper(task_proposer, skills_num, task_name,
                                     normalize(HalfCheetahVelEnv()))
                     for task_name in range(skills_num)]

    env_sampler = EnvPoolSampler(ML_train_envs)
    env = env_sampler.sample(num_train_tasks)

    test_env_sampler = SetTaskSampler(lambda: GarageEnv(normalize(
        HalfCheetahVelEnv())))

    runner = LocalRunner(ctxt)

    qf_env = MetaBasicHierch.get_env_spec(env[0](), num_skills, "qf")

    qf = ContinuousMLPQFunction(env_spec=qf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = MetaBasicHierch.get_env_spec(env[0](), num_skills, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    controller_policy_env = MetaBasicHierch.get_env_spec(env[0](),
                                                  module="controller_policy",
                                                  num_skills=num_skills)

    controller_policy = CategoricalMLPPolicy(env_spec=controller_policy_env,
                                             hidden_sizes=[net_size, net_size],
                                             hidden_nonlinearity=functional.relu)

    meta_basic_hierarchy = MetaBasicHierch(
        env=env,
        skill_env=skill_env,
        controller_policy=controller_policy,
        skill_actor=skill_actor,
        qf=qf,
        vf=vf,
        num_skills=num_skills,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        sampler_class=LocalSkillSampler,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_tasks_sample=num_tasks_sample,
        num_steps_per_epoch=num_steps_per_epoch,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_path_length=max_path_length
    )

    tu.set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        meta_basic_hierarchy.to()

    worker_args = dict(num_skills=num_skills,
                       skill_actor_class=type(skill_actor),
                       controller_class=ControllerPolicy,
                       deterministic=False, accum_context=True)

    runner.setup(algo=meta_basic_hierarchy,
                 env=env[0](),
                 sampler_cls=LocalSkillSampler,
                 sampler_args=dict(max_path_length=max_path_length),
                 n_workers=1,
                 worker_class=BasicHierachWorker,
                 worker_args=worker_args
                 )

    average_returns = runner.train(n_epochs=num_epochs, batch_size=batch_size)
    runner.save(num_epochs - 1)

    return average_returns


returns = meta_basic_hierachy_cheetah_vel()


def save_list_to_file(x, filename):
    with open(filename, 'w') as f:
        for item in x:
            f.write("%s\n" % item)


if not os.path.exists('tmp'):
    os.makedirs('tmp')
save_list_to_file(returns, "tmp/diayn_controller_half_cheetah_returns.txt")
