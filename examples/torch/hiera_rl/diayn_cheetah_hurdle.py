"""An example to test diayn written in PyTorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import garage.torch.utils as tu
from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.envs import normalize
from garage.envs.hierarchical_rl_gym.half_cheetah_hurdle import \
    HalfCheetahEnv_Hurdle
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import SkillWorker
from garage.sampler.local_skill_sampler import LocalSkillSampler
from garage.torch.algos import DIAYN
from garage.torch.algos.discriminator import MLPDiscriminator
from garage.torch.policies import TanhGaussianMLPSkillPolicy
from garage.torch.policies.gaussian_mixture_model import GMMSkillPolicy
from garage.torch.q_functions import ContinuousMLPSkillQFunction


@wrap_experiment(snapshot_mode='gap_and_last', snapshot_gap=100)
def diayn_cheetah_hurdle(ctxt=None, seed=1):

    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(HalfCheetahEnv_Hurdle()))
    skills_num = 6

    # policy = TanhGaussianMLPSkillPolicy(
    #     env_spec=env.spec,
    #     skills_num=skills_num,
    #     hidden_sizes=[300, 300],
    #     hidden_nonlinearity=nn.ReLU,
    #     output_nonlinearity=None,
    #     min_std=np.exp(-20.),
    #     max_std=np.exp(2.),
    # )

    policy = GMMSkillPolicy(
        env_spec=env.spec,
        K=skills_num,
        skills_num=skills_num,
        hidden_layer_sizes=[300, 300])

    qf1 = ContinuousMLPSkillQFunction(env_spec=env.spec,
                                      skills_num=skills_num,
                                      hidden_sizes=[300, 300],
                                      hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPSkillQFunction(env_spec=env.spec,
                                      skills_num=skills_num,
                                      hidden_sizes=[300, 300],
                                      hidden_nonlinearity=F.relu)

    discriminator = MLPDiscriminator(env_spec=env.spec,
                                     skills_num=skills_num,
                                     hidden_sizes=[300, 300],
                                     hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    diayn = DIAYN(env_spec=env.spec,
                  skills_num=skills_num,
                  discriminator=discriminator,
                  policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=1000,
                  max_path_length=1000,
                  replay_buffer=replay_buffer,
                  min_buffer_size=int(1e6),
                  recorded=False,  # enable the video recording func
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
    runner.train(n_epochs=1500, batch_size=1000)


s = np.random.randint(0, 1000)
diayn_cheetah_hurdle(seed=s)  # 521 in the sac_cheetah example
