#!/usr/bin/env python3
"""This is an example to train a task with DQN algorithm in pixel environment.

Here it creates a gym environment Pong, and trains a DQN with 1M steps.
"""
import click
import gym

from garage import wrap_experiment
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteCNNQFunction


@click.command()
@click.option('--buffer_size', type=int, default=int(5e4))
@wrap_experiment
def dqn_pong(ctxt=None, seed=1, buffer_size=int(5e4)):
    """Train DQN on PongNoFrameskip-v4 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        buffer_size (int): Number of timesteps to store in replay buffer.

    """
    set_seed(seed)
    with LocalTFRunner(ctxt) as runner:
        n_epochs = 100
        steps_per_epoch = 20
        sampler_batch_size = 500
        num_timesteps = n_epochs * steps_per_epoch * sampler_batch_size

        env = gym.make('PongNoFrameskip-v4')
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4)

        env = TfEnv(env, is_image=True)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=buffer_size,
                                           time_horizon=1)

        qf = DiscreteCNNQFunction(env_spec=env.spec,
                                  filter_dims=(8, 4, 3),
                                  num_filters=(32, 64, 64),
                                  strides=(4, 2, 1),
                                  dueling=False)

        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
        exploration_policy = EpsilonGreedyPolicy(env_spec=env.spec,
                                                 policy=policy,
                                                 total_timesteps=num_timesteps,
                                                 max_epsilon=1.0,
                                                 min_epsilon=0.02,
                                                 decay_ratio=0.1)

        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_policy=exploration_policy,
                   replay_buffer=replay_buffer,
                   qf_lr=1e-4,
                   discount=0.99,
                   min_buffer_size=int(1e4),
                   double_q=False,
                   n_train_steps=500,
                   steps_per_epoch=steps_per_epoch,
                   target_network_update_freq=2,
                   buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs, batch_size=sampler_batch_size)


dqn_pong()
