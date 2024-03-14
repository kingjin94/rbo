import argparse
from copy import deepcopy
import importlib
from pathlib import Path
import sys
from time import time
from typing import Any, Dict, Optional
from threadpoolctl import ThreadpoolController

import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mcs.optimize.TaskGenerator import TaskGenerator
from timor.utilities import logging

from base_opt.base_opt import config
from base_opt.base_opt.BaseOptimizer import BaseOptimizerBase, BaseOptimizationInfo, str_to_base_optimizer
from base_opt.base_opt.config import add_default_arguments
from base_opt.utilities.Proxies import BaseChangeEnvironment


controller = ThreadpoolController()


@controller.wrap(limits=1, user_api='blas')  # Limit numpy core usage
def evaluate_optimizer(
        optimizer: BaseOptimizerBase, task_generator: TaskGenerator, timeout: float,
        reward_fail: float, action_space: str, trial: Optional[optuna.Trial] = None) -> Dict[str, BaseOptimizationInfo]:
    """
    Evaluate an optimizer on a task generator.

    :param optimizer: The optimizer to evaluate.
    :param task_generator: The task generator to evaluate on.
    :param timeout: The timeout for each task.
    :param reward_fail: The reward for failing a task.
    :param action_space: The action space to use as given by BaseChangeEnvironment.available_action.
    :param trial: The optuna trial to report intermediate results to.
    :return: A tuple of dicts of best actions, best rewards and additional information for each task.
    """
    ret = dict()
    if not hasattr(task_generator, '__len__'):
        logging.warning("Task generator does not have a length. This could take forever.")
    # determine observation space
    observations = set()
    if hasattr(optimizer, 'agent'):
        assert hasattr(optimizer.agent, 'observation_space'), "Expect optimizer with agent to have observation space."
        observations = set(optimizer.agent.observation_space.keys())
        observations.discard('previous')
        assert hasattr(optimizer.agent, 'action_space'), "Expect optimizer with agent to have action space."
        assert BaseChangeEnvironment.available_actions[action_space][1].shape == optimizer.agent.action_space.shape, \
            f"Action space {optimizer.agent.action_space} of agent does not match action space " \
            f"{BaseChangeEnvironment.available_actions[action_space]} of environment."
    if importlib.util.find_spec("ompl") is not None:  # Reduce OMPL logging
        import ompl.util
        ompl.util.setLogLevel(ompl.util.LOG_WARN)
    with logging_redirect_tqdm([logging.getLogger(), ]):
        for step, task in tqdm(enumerate(task_generator.as_finite_iterable()), desc="Tasks"):
            task_solver = config.get_task_solver(task)
            base_change_environment = BaseChangeEnvironment(assembly=config.ASSEMBLY, task_solver=task_solver,
                                                            task=task, reward_fail=reward_fail,
                                                            cost_function=config.COST_FUNCTION,
                                                            observations=observations,
                                                            action2base_pose=action_space,)
            # penalty_ik_distance=1.,  # TODO make configurable
            # penalty_filter_fails=1.)  # TODO make configurable
            # TODO make configurable, e.g., deterministic=False
            ret[task.id] = optimizer.optimize(base_change_environment, timeout=timeout)[2]
            if trial is not None:
                trial.report(np.mean([r.history.best_reward for r in ret.values()]), step=step)
                if trial.should_prune():
                    return ret  # Return intermediate results for logging

    return ret


def evaluate_algorithm(algorithm: str, eval_set: str, timeout: float, reward_fail: float, hps: Dict[str, Any],
                       action_space: str, raw_data: Optional[str] = None,
                       solution_storage: Optional[Path] = None) -> Dict[str, BaseOptimizationInfo]:
    """Evaluate a single algorithm on a single task set."""
    t0 = time()
    ret = evaluate_optimizer(
        str_to_base_optimizer[algorithm](hps, solution_storage=solution_storage),
        config.str_to_task_set[eval_set],
        timeout,
        reward_fail,
        action_space
    )
    print(f"Finished evaluating {algorithm} on task set {eval_set} in {time() - t0} seconds.")
    if raw_data is not None:
        Path(raw_data).parent.mkdir(parents=True, exist_ok=True)
        data = pd.concat([v.to_data_frame(config.SEED) for v in ret.values()], axis=0)
        data.to_csv(raw_data, mode='w', )
        print(f"Raw data stored in {raw_data}.")
        print(f"Additional solutions stored in {solution_storage}.")

    return ret


def calc_summary(ret: Dict[str, BaseOptimizationInfo]) -> pd.DataFrame:
    """Calculate summary statistics from a result dictionary (task ID -> Base Optimizer Info)."""
    success = np.asarray([r.is_success for r in ret.values()])
    frame = pd.DataFrame({
        'Task ID': list(ret.keys()),
        'Rewards': np.asarray([r.history.best_reward for r in ret.values()]),
        'Success': success,
        'First Reward Index': np.asarray([r.history.first_success_idx for r in ret.values()]),
        'First Reward Time': np.asarray([r.history.first_success_time for r in ret.values()]),
        'Actions': [np.asarray(r.history.actions) for r in ret.values()],
        'Steps': [len(r.history) for r in ret.values()],
    })
    return frame


def print_stats(ret: Dict[str, BaseOptimizationInfo]):
    """Print statistics"""
    summary = calc_summary(ret)
    success_rate = summary['Success'].mean()
    print("Mean reward (neg. cost):", summary['Rewards'].mean(), "+/-", summary['Rewards'].std())
    print("Success rate:", success_rate)
    all_actions = np.vstack(summary['Actions'])
    print("Mean action:", all_actions.mean(axis=0), "+/-", all_actions.std(axis=0))
    print("Mean history length ", np.mean([len(r.history.rewards) for r in ret.values()]))
    if success_rate > 0:
        success_rewards = summary.loc[summary['Success']]['Rewards']
        print("Mean reward on successful tasks:", success_rewards.mean(), "+/-", success_rewards.std())
        mean_first_success = summary.loc[summary['Success']]['First Reward Index'].mean()
        print("Mean first successful step (on successful tasks):", mean_first_success)
    print("---------------------------------------------------------------\n\n")
    print("Per task:")
    print(summary[['Task ID', "Rewards", "Success", "Steps", "First Reward Index", "First Reward Time"]].to_string())

    print("---------------------------------------------------------------\n\n")
    print("Failure reasons:")
    df = pd.concat(v.to_data_frame() for v in ret.values())
    print(df['Fail Reason'].value_counts())


def main(*args):
    """Main function to map CLI arguments to arguments for hyperparameter optimization."""
    # Read arguments
    parser = argparse.ArgumentParser(description='Evaluate base optimizer on a set of tasks.')
    parser.add_argument(
        '--hyper-parameters', type=str,
        help='Hyperparameters to use for the optimizer (given in hyperparameters.yml as algorithm: hp_name: hps)',
        default='best')
    parser.add_argument(
        '--raw-data', type=str,
        help='Pickle file to store raw data in (will be overwritten). E.g., sth.csv',
        default=None)
    add_default_arguments(parser)
    args = parser.parse_args(*args)
    assert not args.reward_improvement, "Reward improvement is not supported for evaluation yet."
    assert args.penalty_ik_distance == 0., "Penalty IK distance is not supported for evaluation yet."
    assert args.penalty_filter_fails == 0., "Penalty filter fails is not supported for evaluation yet."

    config.adapt_config(args)
    print(f"Evaluating {args.algorithm} on task set {args.eval_set}.")
    print(f"Arguments: {args}")
    print(f"Hyperparameters: {config.known_hyperparameters[args.algorithm]}")

    # Run hyperparameter optimization
    ret = evaluate_algorithm(algorithm=args.algorithm, eval_set=args.eval_set,
                             timeout=args.timeout, reward_fail=args.reward_fail,
                             hps=deepcopy(config.known_hyperparameters[args.algorithm][args.hyper_parameters]),
                             action_space=args.action_space, raw_data=args.raw_data,
                             solution_storage=args.solution_storage)

    print_stats(ret)


if __name__ == '__main__':
    # logging.setLevel(logging.DEBUG)
    main(sys.argv[1:])
