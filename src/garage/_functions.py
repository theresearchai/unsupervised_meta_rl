"""Functions exposed directly in the garage namespace."""
from collections import defaultdict

from dowel import tabular
import numpy as np

import garage
from garage import TrajectoryBatch
from garage.misc.tensor_utils import discount_cumsum


def log_multitask_performance(itr, batch, discount, trajectory_class=TrajectoryBatch, name_map=None):
    r"""Log performance of trajectories from multiple tasks.

    Args:
        itr (int): Iteration number to be logged.
        batch (garage.TrajectoryBatch): Batch of trajectories. The trajectories
            should have either the "task_name" or "task_id" `env_infos`. If the
            "task_name" is not present, then `name_map` is required, and should
            map from task id's to task names.
        discount (float): Discount used in computing returns.
        name_map (dict[int, str] or None): Mapping from task id's to task
            names. Optional if the "task_name" environment info is present.
            Note that if provided, all tasks listed in this map will be logged,
            even if there are no trajectories present for them.

    Returns:
        numpy.ndarray: Undiscounted returns averaged across all tasks. Has
            shape :math:`(N \bullet [T])`.

    """
    traj_by_name = defaultdict(list)
    for trajectory in batch.split():
        task_name = '__unnamed_task__'
        if 'task_name' in trajectory.env_infos:
            task_name = trajectory.env_infos['task_name'][0]
        elif 'task_id' in trajectory.env_infos:
            name_map = {} if name_map is None else name_map
            task_id = trajectory.env_infos['task_id'][0]
            task_name = name_map.get(task_id, 'Task #{}'.format(task_id))
        traj_by_name[task_name].append(trajectory)
    if name_map is None:
        task_names = traj_by_name.keys()
    else:
        task_names = name_map.values()
    for task_name in task_names:
        if task_name in traj_by_name:
            trajectories = traj_by_name[task_name]
            log_performance(itr,
                            trajectory_class.concatenate(*trajectories),
                            discount,
                            trajectory_class=trajectory_class,
                            prefix=task_name)
        else:
            with tabular.prefix(task_name + '/'):
                tabular.record('Iteration', itr)
                tabular.record('NumTrajs', 0)
                tabular.record('AverageDiscountedReturn', np.nan)
                tabular.record('AverageReturn', np.nan)
                tabular.record('StdReturn', np.nan)
                tabular.record('MaxReturn', np.nan)
                tabular.record('MinReturn', np.nan)
                tabular.record('CompletionRate', np.nan)
                tabular.record('SuccessRate', np.nan)

    return log_performance(itr, batch, discount=discount,
                           trajectory_class=trajectory_class, prefix='Average')


def log_performance(itr, batch, discount, trajectory_class=TrajectoryBatch, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of trajectories.

    Args:
        itr (int): Iteration number.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    completion = []
    success = []
    for trajectory in batch.split():
        if trajectory_class == TrajectoryBatch:
            returns.append(discount_cumsum(trajectory.rewards, discount))
            undiscounted_returns.append(sum(trajectory.rewards))
        else:
            returns.append(discount_cumsum(trajectory.env_rewards, discount))
            undiscounted_returns.append(sum(trajectory.env_rewards))
        completion.append(float(trajectory.terminals.any()))
        if 'success' in trajectory.env_infos:
            success.append(float(trajectory.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumTrajs', len(returns))

        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('CompletionRate', np.mean(completion))
        if success:
            tabular.record('SuccessRate', np.mean(success))

    return undiscounted_returns
