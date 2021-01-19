from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly

# Base class
from garage.envs.hierarchical_rl_gym.base_env import BaseEnv

# Walker2d
from garage.envs.hierarchical_rl_gym.walker2d import Walker2dEnv
#from gym.envs.mujoco.walker2d_v0 import Walker2dEnvV0
from garage.envs.hierarchical_rl_gym.walker2d_forward import Walker2dForwardEnv
from garage.envs.hierarchical_rl_gym.walker2d_backward import Walker2dBackwardEnv
from garage.envs.hierarchical_rl_gym.walker2d_balance import Walker2dBalanceEnv
from garage.envs.hierarchical_rl_gym.walker2d_jump import Walker2dJumpEnv
from garage.envs.hierarchical_rl_gym.walker2d_crawl import Walker2dCrawlEnv
from garage.envs.hierarchical_rl_gym.walker2d_patrol import Walker2dPatrolEnv
from garage.envs.hierarchical_rl_gym.walker2d_hurdle_v0 import Walker2dHurdleEnvV0
from garage.envs.hierarchical_rl_gym.walker2d_hurdle import Walker2dHurdleEnv
from garage.envs.hierarchical_rl_gym.walker2d_obstacle_course import Walker2dObstacleCourseEnv

# Jaco
from garage.envs.hierarchical_rl_gym.jaco import JacoEnv
from garage.envs.hierarchical_rl_gym.jaco_pick import JacoPickEnv
from garage.envs.hierarchical_rl_gym.jaco_catch import JacoCatchEnv
from garage.envs.hierarchical_rl_gym.jaco_toss import JacoTossEnv
from garage.envs.hierarchical_rl_gym.jaco_hit import JacoHitEnv
from garage.envs.hierarchical_rl_gym.jaco_keep_pick import JacoKeepPickEnv
from garage.envs.hierarchical_rl_gym.jaco_keep_catch import JacoKeepCatchEnv
from garage.envs.hierarchical_rl_gym.jaco_serve import JacoServeEnv

# HalfCheetahEnv
from garage.envs.hierarchical_rl_gym.half_cheetah import HalfCheetahEnv
from garage.envs.hierarchical_rl_gym.half_cheetah_hurdle import HalfCheetahEnv_Hurdle
from garage.envs.hierarchical_rl_gym.half_cheetah_hurdle_v2 import HalfCheetahEnv_Hurdle_v2
from garage.envs.hierarchical_rl_gym.half_cheetah_hurdle_v3 import HalfCheetahEnv_Hurdle_v3
