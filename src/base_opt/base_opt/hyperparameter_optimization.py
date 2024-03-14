import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import sys
from time import process_time, time
from typing import Any, Callable, Dict, List, Type
from threadpoolctl import ThreadpoolController

import numpy as np
import optuna
import pandas as pd

from mcs.optimize.TaskGenerator import TaskGenerator
from timor.utilities import logging

from base_opt.base_opt import config
from base_opt.base_opt.evaluate_algorithm import evaluate_optimizer
from base_opt.base_opt.hyperparameter_search_spaces import algorithm_to_search_space
from base_opt.base_opt.BaseOptimizer import BaseOptimizerBase, str_to_base_optimizer


controller = ThreadpoolController()


def evaluate_trial(trial: optuna.Trial, optimizer_class: Type[BaseOptimizerBase],
                   search_space: Callable[[optuna.Trial], Dict[str, Any]], timeout: float, reward_fail: float,
                   task_generator: TaskGenerator, action_space: str):
    """Evaluate a single trial."""
    trial.set_user_attr('pid', os.getpid())  # E.g. to debug heavy CPU usage
    t0 = process_time()
    optimizer = optimizer_class(search_space(trial))  # No need to save solution
    ret = evaluate_optimizer(optimizer, task_generator, timeout, reward_fail, action_space, trial)
    trial.set_user_attr('process_time', process_time() - t0)  # Measure processor usage time for trial
    if len(ret) > 0:
        trial.set_user_attr('success_rate', np.mean([r.is_success for r in ret.values()]))
        detailed_results_file = config.TMP_DIR.joinpath(f"trial_{trial.number}.csv")
        pd.concat([r.to_data_frame(config.SEED) for r in ret.values()]).to_csv(detailed_results_file)
        trial.set_user_attr('detailed_results', str(detailed_results_file))
        trial.set_user_attr('average_steps', np.mean([len(r.history) for r in ret.values()]))
    if trial.should_prune():
        raise optuna.TrialPruned()  # Already contains mean best reward as intermediate result
    return np.mean([r.history.best_reward for r in ret.values()])  # Return average reward as objective


@controller.wrap(limits=1, user_api='blas')  # Limit numpy core usage
def evaluate_n_algorithm_hps(study: optuna.study,
                             n_trials: int,
                             optimizer_class: Type[BaseOptimizerBase],
                             search_space: Callable[[optuna.Trial], Dict[str, Any]],
                             timeout: float, reward_fail: float, task_generator: TaskGenerator,
                             action_space: str):
    """Create and check a single hyperparameter configuration."""
    counted_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.PRUNED
    }
    study.sampler.reseed_rng()  # Force reseed s.t. parallel trials are independent

    while len(study.get_trials(counted_states)) < n_trials:
        study.optimize(
            lambda trial: evaluate_trial(trial, optimizer_class, search_space, timeout, reward_fail, task_generator,
                                         action_space),
            n_jobs=1,  # Single job s.t. process_time correctly measures time
            callbacks=[
                optuna.study.MaxTrialsCallback(n_trials=n_trials, states=counted_states),
            ]
        )


def hp_tune(algorithm: str, n_trials: int, parallel_trials: int, storage: str, sampler: str, pruner: str,
            study_name: str, seed: int, device: str, timeout: float, reward_fail: float, penalty_ik_distance: float,
            penalty_filter_fails: float, train_set: str, eval_set: str, observations: List[str], action_space: str,
            optimize_hyperparameters: bool, rl_algorithm: str, reward_improvement: bool, additional_args: List[str]):
    """Run hyperparameter optimization for a single algorithm."""
    t0 = time()

    if optimize_hyperparameters:
        # Create study
        if 'sqlite://' in storage and Path(storage[10:]).is_file() and not os.access(storage[10:], os.W_OK):
            logging.warning(f"Storage {storage} might be unwritable... Still trying but might fail.")
        storage = optuna.storages.RDBStorage(url=storage,
                                             engine_kwargs={"connect_args": {"timeout": 100}})
        study = optuna.create_study(sampler=config.create_sampler(sampler), pruner=config.create_pruner(pruner),
                                    storage=storage, study_name=study_name, load_if_exists=True, direction='maximize')
        study_name = study.study_name  # Get actual study name, e.g., random if None given

        # Store study meta-data
        study.set_user_attr('algorithm', algorithm)
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            study.set_user_attr('git_hash', git_hash)
        except subprocess.CalledProcessError:
            logging.warning("Could not get git hash to save inside study.")
        study.set_user_attr('seed', seed)
        study.set_user_attr('timeout', timeout)
        study.set_user_attr('reward_fail', reward_fail)
        study.set_user_attr('penalty_ik_distance', penalty_ik_distance)
        study.set_user_attr('penalty_filter_fails', penalty_filter_fails)
        study.set_user_attr('train_set', train_set)
        study.set_user_attr('eval_set', eval_set)
        study.set_user_attr('observations', observations)
        study.set_user_attr('action_space', action_space)
        study.set_user_attr('rl_algorithm', rl_algorithm)
        study.set_user_attr('additional_args', additional_args)

    if len(additional_args) > 0:
        logging.warning(f"Additional arguments {additional_args} are ignored for {algorithm}.")
    optimizer_class = str_to_base_optimizer[algorithm]
    optimizer_search_space = algorithm_to_search_space[algorithm]
    if train_set is not None:
        logging.warning(f"Train set {train_set} is ignored for {algorithm}.")
    os.environ['OMP_NUM_THREADS'] = '4'  # Limit numpy to sensible number of threads
    if parallel_trials > 1:
        with Pool(parallel_trials) as pool:
            pool.starmap(evaluate_n_algorithm_hps,
                         [(study, n_trials, optimizer_class, optimizer_search_space, timeout, reward_fail,
                           config.str_to_task_set[eval_set], action_space) for _ in range(parallel_trials)])
    else:
        evaluate_n_algorithm_hps(
            study, n_trials, optimizer_class, optimizer_search_space, timeout, reward_fail,
            config.str_to_task_set[eval_set], action_space)

    if optimize_hyperparameters:
        # Summarize results
        print(f"Best hyperparameters for {algorithm}:")
        print(study.best_params)

        # Add warnings if not all run, timeout, ...
        if len(study.get_trials(states=[optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.COMPLETE])) < n_trials:
            print(f"Only {len(study.trials)} trials run instead of {n_trials}!")
            print(f"Trials and their states: \n{study.trials}")
        print(f"Hyperparameter optimization took {time() - t0} seconds.")
    else:
        print("Trained agent!")


def main(*args):
    """Main function to map CLI arguments to arguments for hyperparameter optimization."""
    # Read arguments
    parser = argparse.ArgumentParser(
        description='''
        Run hyperparameter optimization for different base optimizers.
        Additional arguments are passed to rl_zoo3.train
        ''',
    )
    config.add_default_arguments(parser)
    parser.add_argument("--n-trials", type=int, help="Number of trials to run", default=100)
    parser.add_argument("--parallel-trials", type=int, help="Number of parallel trials to run", default=1)
    parser.add_argument("--storage", type=str,
                        help="Location of study database, e.g., sqlite:///./hyper_opt.db"
                             "For details see Optuna's storage documentation",
                        default="sqlite:///./hyper_opt.db")
    parser.add_argument("--study-name", type=str, help="Postfix to add to study name", default=None)
    parser.add_argument("--sampler", type=str, help="Optuna sampler to use", default="tpe",
                        choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', type=str, help='Optuna pruner to use', default='median',
                        choices=['none', 'median', 'halving'])
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true", default=False,
                        help="Run hyperparameter optimization (can only be deactivated for RL)")

    args, additional_args = parser.parse_known_args(*args)

    if args.algorithm in {'RandomBaseOptimizer', 'DummyBaseOptimizer'}:
        print(f"Optimizer {args.algorithm} has not hyperparameters to optimize.")
        return
    if not args.optimize_hyperparameters and args.algorithm != 'RLOptimizer':
        raise ValueError(f"Algorithm {args.algorithm} cannot be besides running hyperparameter optimization.")
    config.adapt_config(args)

    # Run hyperparameter optimization
    hp_tune(args.algorithm, args.n_trials, args.parallel_trials, args.storage,
            args.sampler, args.pruner, args.study_name, args.seed, args.device, args.timeout, args.reward_fail,
            args.penalty_ik_distance, args.penalty_filter_fails, args.train_set, args.eval_set, args.observations,
            args.action_space, args.optimize_hyperparameters, args.rl_algorithm, args.reward_improvement,
            additional_args)


if __name__ == '__main__':
    main(sys.argv[1:])
