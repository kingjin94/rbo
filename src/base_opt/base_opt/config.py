import argparse
from argparse import ArgumentParser
from pathlib import Path
import tempfile

import numpy as np
import optuna
import yaml

from mcs.TaskSolver import SimpleHierarchicalTaskSolverWithoutBaseChange, TaskSolverBase
from mcs.utilities.default_robots import get_six_axis_modrob_v2
from timor.configuration_search.AssemblyFilter import AssemblyModuleLengthFilter, InverseKinematicsSolvable, \
    RobotCreationFilter
from timor.task import CostFunctions
from timor.task.Task import Task

from base_opt.utilities.AssemblyFilter import RobotLongEnoughFilter
from base_opt.utilities.Proxies import BaseChangeEnvironment
from base_opt.utilities.TaskGenerator import FixedSetTaskGenerator

# Configure Random Number Generators
SEED = 42
RNG = np.random.default_rng(SEED)
REWARD_FAIL = -20.0
TMP_DIR = None


# Configure Robot Assemblies
assemblies = {
    'modrob-gen2': get_six_axis_modrob_v2()
}
ASSEMBLY = assemblies['modrob-gen2']


# Configure Task Files and Task Generators
EVAL_TASK_FILES = [Path(f'data/cobra_cache/tasks/2022/Whitman2020/with_torque/variable_base_xyz_any_rot/3g_3o/{i}.json')
                   for i in range(70)]
EVAL_MIN_FILES = EVAL_TASK_FILES[5:8]  # Small subset for testing and debugging; 6 easy to solve
TEST_SIMPLE_TASK_FILES = \
    [Path(f'data/cobra_cache/tasks/2022/Whitman2020/with_torque/variable_base_xyz_any_rot/3g_3o/{i}.json')
     for i in range(70, 100)]
TEST_HARD_TASK_FILES = \
    [Path(f'data/cobra_cache/tasks/2022/Whitman2020/with_torque/variable_base_xyz_any_rot/5g_5o/{i}.json')
     for i in range(100)]
TEST_REALWORLD_TASK_FILES = \
    [p for p in Path('data/cobra_cache/tasks/2022/').rglob('Liu2020/Case2b/variable_base_xyz_any_rot/*.json')]

EVAL_TASK_GENERATOR = FixedSetTaskGenerator(
    rng=RNG,
    tasks=[Task.from_json_file(f) for f in EVAL_TASK_FILES])
EVAL_MIN_TASK_GENERATOR = FixedSetTaskGenerator(
    rng=RNG,
    tasks=[Task.from_json_file(f) for f in EVAL_MIN_FILES])
TEST_SIMPLE_TASK_GENERATOR = FixedSetTaskGenerator(
    rng=RNG,
    tasks=[Task.from_json_file(f) for f in TEST_SIMPLE_TASK_FILES])
TEST_HARD_TASK_GENERATOR = FixedSetTaskGenerator(
    rng=RNG,
    tasks=[Task.from_json_file(f) for f in TEST_HARD_TASK_FILES])
TEST_REALWORLD_TASK_GENERATOR = FixedSetTaskGenerator(
    rng=RNG,
    tasks=[Task.from_json_file(f) for f in TEST_REALWORLD_TASK_FILES])


str_to_task_set = {
    "eval": EVAL_TASK_GENERATOR,
    "eval_min": EVAL_MIN_TASK_GENERATOR,
    "test_simple": TEST_SIMPLE_TASK_GENERATOR,
    "test_hard": TEST_HARD_TASK_GENERATOR,
    "test_realworld": TEST_REALWORLD_TASK_GENERATOR,
}


# Configure Cost Function
COST_FUNCTION = CostFunctions.CycleTime()

# Create solver
TASK_SOLVER_TIMEOUT = 10.0


def assembly_filters(task: Task):
    """Create assembly filters for a given task; tweaked from default_filters in timor.AssemblyFilter."""
    assembly_length = AssemblyModuleLengthFilter()
    create_robot = RobotCreationFilter()
    robot_length_filter = RobotLongEnoughFilter()
    ik_simple = InverseKinematicsSolvable(ignore_self_collisions=True, max_iter=300)  # Increase to reduce rejects
    ik_complex = InverseKinematicsSolvable(task=task, max_iter=1500)
    return assembly_length, create_robot, robot_length_filter, ik_simple, ik_complex


def get_task_solver(task: Task) -> TaskSolverBase:
    """Create task solver for a given task."""
    global ASSEMBLY
    return SimpleHierarchicalTaskSolverWithoutBaseChange(
        task,
        assembly=ASSEMBLY,
        filters=assembly_filters(task),
        timeout=TASK_SOLVER_TIMEOUT,
        ik_generation_kwargs={"min_ik_count": 1,  # Reduce time spent on finding IK candidates
                              "max_ik_count": 3,
                              "minimum_search_time": 0.2},
        solve_goals_sequentially=True)  # Only optimize connect-ability not overall run-time yet


def create_sampler(sampler: str, *, n_startup_trials: int = 10):
    """
    Create optuna sampler from string.

    :source: rl_zoo3.exp_manager
    """
    if sampler == 'random':
        return optuna.samplers.RandomSampler(SEED)
    if sampler == 'tpe':
        return optuna.samplers.TPESampler(seed=SEED, multivariate=True,
                                          n_startup_trials=n_startup_trials)
    if sampler == 'skopt':
        return optuna.integration.SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    raise ValueError(f"Unknown sampler {sampler}")


def create_pruner(pruner: str, *, n_startup_trials: int = 10, n_warmup_steps: int = 10):
    """
    Create optuna pruner from string.

    :source: rl_zoo3.exp_manager
    """
    if pruner == 'none':
        return optuna.pruners.NopPruner()
    if pruner == 'median':
        return optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
    if pruner == 'halving':
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    raise ValueError(f"Unknown pruner {pruner}")


def set_seed(seed: int):
    """Set seed for random number generator."""
    global RNG, SEED
    SEED = seed
    RNG = np.random.default_rng(seed)


def adapt_config(args):
    """Adapt config to arguments."""
    set_seed(args.seed)
    global ASSEMBLY
    ASSEMBLY = assemblies[args.assembly]
    global TASK_SOLVER_TIMEOUT
    TASK_SOLVER_TIMEOUT = args.task_solver_timeout
    global COST_FUNCTION
    COST_FUNCTION = CostFunctions.abbreviations[args.cost_function]()
    if args.solution_storage is not None:
        args.solution_storage = Path(args.solution_storage)
        args.solution_storage.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is None:
        args.tmp_dir = tempfile.TemporaryDirectory().name
    args.tmp_dir = Path(args.tmp_dir)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    global TMP_DIR
    TMP_DIR = args.tmp_dir


def add_default_arguments(parser: ArgumentParser):
    """Add default arguments for hyperparameter opt, evaluation to an argument parser."""
    from base_opt.base_opt.BaseOptimizer import str_to_base_optimizer  # Avoid circular import
    parser.add_argument("--algorithm", type=str, help='Base optimizer algorithm to use',
                        choices=str_to_base_optimizer.keys())
    parser.add_argument("--eval-set", type=str,
                        help='Task set to use for testing algorithm/evaluating hyperparameters.',
                        choices=str_to_task_set.keys(),
                        default="eval")
    parser.add_argument("--timeout", type=float, help='Timeout for optimizing a single task´s base pose',
                        default=30.0)
    parser.add_argument("--reward-fail", type=float,
                        help='Timeout for optimizing a single task´s base pose',
                        default=REWARD_FAIL)
    parser.add_argument("--assembly", type=str, help='Assembly to use', choices=assemblies.keys(),
                        default="modrob-gen2")
    parser.add_argument("--action-space", type=str, help='Action space to use',
                        choices=BaseChangeEnvironment.available_actions, default="xyz")
    parser.add_argument("--seed", type=int, help='Seed to use for random number generator', default=42)
    parser.add_argument("--task-solver-timeout", type=float, help='Timeout for task solver',
                        default=10.)
    parser.add_argument("--cost-function", type=str, help='Cost function to optimize', default="cyc",
                        choices=CostFunctions.abbreviations.keys())
    parser.add_argument("--solution-storage", type=Path, help='Path to store found solutions in',
                        default=None)
    parser.add_argument("--tmp-dir", type=Path,
                        help='Temporary directory to store intermediate results in s.a. evaluation scores',
                        default=None)
    parser.add_argument("--reward-improvement", action=argparse.BooleanOptionalAction,
                        help='If true the reward is the difference to the previous best', default=False)
    parser.add_argument("--penalty-ik-distance", type=float, default=0.,
                        help='Penalty for IK distance; only used by RLOptimizer during training')
    parser.add_argument("--penalty-filter-fails", type=float, default=0.,
                        help='Penalty for number of failed/non-tested filters')


with open(Path(__file__).parent.joinpath('hyperparameters.yml'), 'r') as f:
    known_hyperparameters = yaml.safe_load(f)
