"""An example to test diayn written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.envs.hierachical_rl.multigoal import MultiGoalEnv
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import SkillWorker
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos import DIAYN
from garage.torch.algos.discriminator import MLPDiscriminator
from garage.torch.policies import TanhGaussianMLPSkillPolicy
from garage.torch.q_functions import ContinuousMLPSkillQFunction


@wrap_experiment(snapshot_mode='gap_and_last', snapshot_gap=100)
def diayn_point_mass_multigoal(ctxt=None, seed=1):

    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = MultiGoalEnv()
    skills_num = 6

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
                  max_path_length=500,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1e4,
                  recorded=True,  # enable the video recording func
                  is_gym_render=False,
                  media_save_path='diayn_2d_multigoal/',
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
    runner.train(n_epochs=1000, batch_size=1000)


s = np.random.randint(0, 1000)
diayn_point_mass_multigoal(seed=s)  # 521 in the sac_cheetah example
