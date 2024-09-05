import abc
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from time import process_time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd
from pygad import pygad
import skopt
import torch.optim

from mcs.optimize.Sampler import GridSampler
from timor import Transformation
from timor.task.Solution import SolutionTrajectory

from base_opt.base_opt import config
from base_opt.utilities.Proxies import BaseChangeEnvironment


@dataclass
class BaseOptimizationHistory:
    """History of an optimization run."""

    _reward_fail: float = field(default=-np.inf, init=True)
    actions: List[np.ndarray] = field(default_factory=list, init=False)
    rewards: List[float] = field(default_factory=list, init=False)
    solutions: List[Optional[uuid.UUID]] = field(default_factory=list, init=False)
    valid_solutions: List[bool] = field(default_factory=list, init=False)
    run_times: List[float] = field(default_factory=list, init=False)
    fail_reasons: List[Optional[str]] = field(default_factory=list, init=False)

    def add_step(self, action: np.ndarray, reward: float, info: Dict[str, Any],
                 run_time: float, solution_storage: Optional[Path] = None):
        """
        Add a step to the history.

        :param action: The action taken.
        :param reward: The reward received.
        :param info: Additional information about the step. Expected to be from mcs.environments.Proxies. Esp:
          * is_success: Whether the task was solved.
          * solution: The solution trajectory if provided.
          * failure modes, s.a., failed_filters, path_planning_failed, trajectory_generation_failed, timeout_task_solver
        :param run_time: The time after the start of the optimization run at the end of the step.
        :param solution_storage: Where to store created solutions.
          If not provided, solutions are not stored or remembered.
        """
        self.actions.append(action)
        self.rewards.append(reward)
        if 'solution' in info:
            solution = info['solution']
            self.valid_solutions.append(solution.valid if solution is not None else False)
            if solution is not None and solution_storage is not None:
                sol_uuid = uuid.uuid4()
                solution.to_json_file(Path(solution_storage).joinpath(f"solution-{sol_uuid}.json"))
                self.solutions.append(sol_uuid)
            else:
                self.solutions.append(None)
        else:
            self.valid_solutions.append(info['is_success'])
            self.solutions.append(None)
        self.run_times.append(run_time)
        # Parse fail reason
        if info['is_success']:
            self.fail_reasons.append(None)
        elif 'failed_filters' in info:
            self.fail_reasons.append(f"Failed filters: {info['failed_filters']}")
        elif info.get('path_planning_failed', False):
            self.fail_reasons.append("Path planning failed")
        elif info.get('trajectory_generation_failed', False):
            self.fail_reasons.append("Trajectory generation failed")
        elif info.get('timeout_task_solver', False):
            self.fail_reasons.append("Timeout task solver")
        else:
            self.fail_reasons.append(f"Unknown; info: {info}")

    @property
    def any_valid(self) -> bool:
        """Return whether any solution was valid."""
        return any(self.valid_solutions)

    @property
    def first_success_idx(self) -> Optional[int]:
        """Return index of first successful step."""
        try:
            return self.valid_solutions.index(True)
        except ValueError:
            return None

    @property
    def first_success_time(self) -> Optional[float]:
        """Return time of first successful step."""
        idx = self.first_success_idx
        return self.run_times[idx] if idx is not None else None

    @property
    def best_solution(self) -> Optional[uuid.UUID]:
        """Return the UUID of the best solution."""
        if self.any_valid:
            return self.solutions[self._best_reward_idx]
        return None

    @property
    def _best_reward_idx(self) -> Optional[int]:
        """Return the index of the best solution."""
        if self.any_valid:
            return np.argmax(self.rewards)
        return None

    @property
    def best_reward(self) -> float:
        """Return the best reward."""
        if len(self) == 0:
            return self._reward_fail
        return max(self.rewards)

    @property
    def best_valid_reward(self) -> float:
        """Return the reward of the best valid solution."""
        if self.any_valid:
            return max(self.rewards)
        return self._reward_fail

    @property
    def best_action(self) -> Optional[np.ndarray]:
        """Return the best action."""
        idx = self._best_reward_idx
        return self.actions[idx] if idx is not None else None

    @property
    def reward_fail(self) -> float:
        """Return the reward for failing to solve the task."""
        return self._reward_fail

    def to_data_frame(self) -> pd.DataFrame:
        """Turn the optimization history into a pandas DataFrame."""
        return pd.DataFrame({
            "Action": self.actions,
            "Reward": self.rewards,
            "Solution": self.solutions,
            "Valid Solution": self.valid_solutions,
            "Run Time": self.run_times,
            "Fail Reason": self.fail_reasons,
            "Reward Fail": self._reward_fail},
            index=pd.Index(range(len(self)), name="Step"))  # TODO Add index = step

    def __len__(self):
        """Return the number of steps in the history."""
        return len(self.actions)


@dataclass
class BaseOptimizationInfo:
    """Additional Information about the optimization process."""

    optimizer_spec: Dict[str, Any]
    optimizer_runtime: float
    task_id: str  # Task ID
    is_success: bool  # Whether the optimization was successful
    action_to_pose: str  # Action to pose function
    history: BaseOptimizationHistory = field(default_factory=BaseOptimizationHistory)
    best_base_pose: Optional[Transformation] = None
    best_solution: Optional[SolutionTrajectory] = None  # Store UUID only

    def to_data_frame(self, seed: Optional[int] = None) -> pd.DataFrame:
        """Turn the optimization info into a pandas DataFrame."""
        run_data = self.history.to_data_frame()
        run_data["Optimizer Runtime"] = self.optimizer_runtime
        run_data["Task ID"] = self.task_id
        run_data["Algorithm"] = self.optimizer_spec["alg"]
        run_data["Optimizer Spec"] = json.dumps(self.optimizer_spec)
        run_data["Seed"] = seed
        run_data["Success"] = self.is_success
        return run_data


class BaseOptimizerBase:
    """A class trying to find the best action in a given BaseChangeEnvironment."""

    best_hps = {}  # Dictionary of best hyperparameters for each optimizer based on hyperparameter optimization

    def __init__(self, hp: Dict, solution_storage: Optional[Path] = None):
        """Initialize the optimizer with the given hyperparameters."""
        self.solution_storage = solution_storage
        self.optimization_history = BaseOptimizationHistory()
        self._t0 = None  # Time of last optimization run start
        self._best_reward = -np.inf
        self._best_action = None
        self._best_info = {}
        self._env = None
        self._timeout = np.inf
        self._reset()

    @property
    def store_solutions(self) -> bool:
        """Return whether to store solutions."""
        return self.solution_storage is not None

    @property
    def time_left(self) -> bool:
        """Return time left for optimization."""
        return process_time() - self._t0 < self._timeout

    @classmethod
    def from_specs(cls, specs: Dict):
        """Initialize the optimizer from a dictionary of specs."""
        alg = specs.pop("alg", None)
        if alg not in str_to_base_optimizer:
            raise NotImplementedError(f"Unknown optimizer {alg}")
        return str_to_base_optimizer[alg](specs)

    @property
    @abc.abstractmethod
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        pass

    @abc.abstractmethod
    def _optimize(self):
        """Implement the optimization algorithm loop in here."""

    def optimize(self, env: BaseChangeEnvironment,
                 timeout: float = 10.0) -> Tuple[np.ndarray, float, BaseOptimizationInfo]:
        """
        Optimize the base of the assembly in the given environment.

        :param env: The base change environment to optimize the base pose in
        :param timeout: The timeout in seconds
        :return: Optimized base, reward, info dictionary
        """
        self._env = env
        self._timeout = timeout

        self._reset()
        self._optimize()
        return self._finalize()

    def _evaluate(self, action: np.ndarray) -> float:
        """Evaluate the action in the environment. Keeps track of the best action and reward."""
        if not self.time_left:
            return self._env.reward_fail  # Shortcut to stop optimization
        obs, reward, done, truncated, info = self._env.step(action, detailed_info=self.store_solutions)
        self._remember_step(action, reward, info)
        return reward

    def _remember_step(self, action, reward, info):
        """Add action, reward and info to the optimization history; update best buffers."""
        if reward > self._best_reward and info['is_success']:
            self._best_reward, self._best_action, self._best_info = reward, action, info
        self.optimization_history.add_step(
            action, reward, info,
            process_time() - self._t0,
            self.solution_storage)

    def _finalize(self) -> Tuple[np.ndarray, float, BaseOptimizationInfo]:
        """Finalize the optimization and return the best action, reward and info."""
        return self._best_action, self._best_reward, BaseOptimizationInfo(
            self.specs, process_time() - self._t0, self._env.task.id, self._best_info.get('is_success', False),
            self._env.action2base_pose, self.optimization_history,
            self._env.action2base_pose(self._best_action) if self._best_action is not None else None,
            self._best_info.get("solution", None) if self.store_solutions else None)

    def _reset(self):
        """Reset the optimizer for a new optimization run; call at start of optimize."""
        self._best_info = {}
        self._best_reward = self._env.reward_fail if self._env is not None else -np.inf
        self.optimization_history = BaseOptimizationHistory(self._best_reward)
        self._best_action = None
        self._t0 = process_time()


class RandomBaseOptimizer(BaseOptimizerBase):
    """Randomly try actions from the environment's action space."""

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        return {"alg": "RandomBaseOptimizer"}

    def _next_action(self, env: BaseChangeEnvironment) -> np.ndarray:
        """Return the next action to take."""
        return env.action_space.sample()

    def _reset_next_action(self, env: BaseChangeEnvironment):
        """Reset internals for next optimization trial."""
        pass

    def _optimize(self):
        """Randomly samples base poses and returns the best one."""
        self._reset_next_action(self._env)
        while self.time_left:
            self._env.reset()  # Reset as optimizer cannot learn from multistep anyway
            try:
                action = self._next_action(self._env)
            except StopIteration:
                break
            self._evaluate(action)


class GradientOptimizer(RandomBaseOptimizer):
    """Gradient optimizer that follows the gradient towards a locally good robot base pose."""

    def __init__(self, hp: Dict, solution_storage: Optional[Path] = None):
        """
        Initialize with number of local search steps and IK steps to take.

        :param hp: Dict of hyperparameters for the optimizer.
                     * local_search_steps: Number of local search steps to take to optimize random guess (default 10).
                     * local_ik_steps: Number of inverse kinematics (IK) steps to take in each local search step
                                       (default 100). With this the joint angles are adapted to the changed base pose
                                       to minimize the distance to the goal poses.
        """
        super().__init__(hp, solution_storage)
        self._local_search_steps = hp.get("local_search_steps", 10)
        self._local_ik_steps = hp.get("local_ik_steps", 100)

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        return {"alg": "GradientOptimizer",
                "local_search_steps": self._local_search_steps,
                "local_ik_steps": self._local_ik_steps}

    def _next_action(self, env: BaseChangeEnvironment) -> np.ndarray:
        """Return the next action to take."""
        guess = env.action_space.sample()
        return self._improve_guess(env, guess)

    @abc.abstractmethod
    def _take_local_step(self, env: BaseChangeEnvironment, T_closest: Dict[str, Transformation],
                         guess: np.ndarray) -> np.ndarray:
        """
        Take a local step towards a better robot base pose; return that better action.

        :param env: The environment to optimize in.
        :param T_closest: The closest end-effector poses found for each goal for the current guess with inv. kin.
        :param guess: The current guess for the action (= parameters of base pose) to take.
        """
        pass

    def _improve_guess(self, env: BaseChangeEnvironment, guess: np.ndarray) -> np.ndarray:
        """
        Improve the guess by taking local steps towards a better robot base pose.
        """
        robot = env.task_solver.robot
        q_closest = {}
        for _ in range(self._local_search_steps):
            guess_base = env.action2base_pose(guess)
            robot.set_base_placement(guess_base)
            q_closest = {g.id: robot.ik(g.goal_pose, max_iter=self._local_ik_steps,
                                        q_init=q_closest[g.id][0] if g.id in q_closest else None)
                         for g in env.task.goals}
            if all(ik_res[1] for ik_res in q_closest.values()):  # Short circuit if all goals are already solved
                return guess
            T_closest = {g_id: guess_base.inv @ robot.fk(q) for g_id, (q, _) in q_closest.items()}
            guess = self._take_local_step(env, T_closest, guess)
        return guess


class AdamOptimizer(GradientOptimizer):
    """
    Use adam for choosing local steps relative to gradient.

    :cite: https://arxiv.org/abs/1412.6980
    """

    best_hps = config.known_hyperparameters["AdamOptimizer"]["best"]

    def __init__(self, hp: Dict, solution_storage: Optional[Path] = None,
                 dtype: torch.dtype = torch.float32):
        """Initialize with learning rate (and dtype) for the adam optimizer."""
        self.hp = deepcopy(self.best_hps)
        super().__init__(self.hp, solution_storage)
        self.hp.update(hp)
        self._dtype = dtype

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        ret = super().specs
        ret.update({"alg": "AdamOptimizer", **self.hp})
        return ret

    def _reset_next_action(self, env: BaseChangeEnvironment):
        self._param = torch.tensor(env.action_space.sample(), requires_grad=True)
        self._opt = torch.optim.Adam(
            [self._param, ], lr=self.hp["lr"], betas=(self.hp["beta1"], self.hp["beta2"]))
        self._env = env

    def _tensor_ik_cost_function(self, action: torch.Tensor, eef_is: torch.Tensor, eef_goal: torch.Tensor):
        """PyTorch Version of the default IK cost function in timor.robot.Robot."""
        base = self._env.action2base_pose(action)
        eef = base @ eef_is  # Pose of end-effector with this base pose
        eef_inv = torch.eye(4, dtype=self._dtype)
        eef_inv[:3, :3] = eef[:3, :3].t()  # Inverse of rotation is transposed
        eef_inv[:3, 3] = -eef[:3, :3].t() @ eef[:3, 3]  # Inverse of end-effector position
        delta = eef_inv @ eef_goal
        translation_error = delta[:3, 3].norm(dim=-1)
        if delta[:3, :3].trace() > 2.999:  # cut off boundaries of rotation
            rotation_error = 0.
        elif delta[:3, :3].trace() < -0.9999:  # cut off boundaries of rotation
            rotation_error = np.pi
        else:
            rotation_error = torch.acos((delta[:3, :3].trace() - 1) / 2)  # angle of rotation via Rodrigues formula
        translation_weight = 1.  # Same as timor.robot.Robot.default_ik_cost_function
        rotation_weight = .5 / np.pi
        return (translation_weight * translation_error + rotation_weight * rotation_error) / \
            (translation_weight + rotation_weight)

    def _take_local_step(self, env: BaseChangeEnvironment, T_closest: Dict[str, Transformation],
                         guess: np.ndarray) -> np.ndarray:
        """
        Use adam to take a local step towards a better robot base pose.
        """
        self._param.data = torch.tensor(guess, dtype=self._dtype)  # Set initial guess
        self._opt.zero_grad()
        with torch.autograd.detect_anomaly():  # Detect any nan or inf in gradients
            # Calculate cost for each goal
            tmp_cost = {g_id: self._tensor_ik_cost_function(
                        self._param,
                        torch.tensor(T.homogeneous, dtype=self._dtype),
                        torch.tensor(
                            env.task.goals_by_id[g_id].goal_pose.nominal.in_world_coordinates().homogeneous,
                            dtype=self._dtype))
                        for g_id, T in T_closest.items()}
            mean_cost = torch.mean(torch.stack(list(tmp_cost.values())))
            mean_cost.backward(retain_graph=True, create_graph=True)
        self._opt.step()
        return self._param.detach().numpy().clip(env.action_space.low, env.action_space.high)


class DummyOptimizer(RandomBaseOptimizer):
    """Always return central action."""

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        return {"alg": "DummyOptimizer"}

    def _next_action(self, env: BaseChangeEnvironment) -> np.ndarray:
        """Return the next action to take - the center of the action space."""
        return (env.action_space.high + env.action_space.low) / 2


class RandomGrid(RandomBaseOptimizer):
    """Choose random values from a grid."""

    def __init__(self, hp: Dict, solution_storage: Optional[Path] = None):
        """Initialize with grid of step_width in each of the action space's dimensions."""
        super().__init__(hp, solution_storage)
        self.num_steps = hp.get("num_steps", 5)
        self.iterator = None

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        return {"alg": "RandomGrid", "num_steps": self.num_steps}

    def _reset_next_action(self, env: BaseChangeEnvironment):
        self.sampler = GridSampler(
            np.tile(self.num_steps, env.action_space.shape), 2 / self.num_steps)
        self.iterator = self.sampler.as_finite_iterable(randomize=True)

    def _next_action(self, env: BaseChangeEnvironment) -> np.ndarray:
        """Return the next action to take."""
        return next(self.iterator)


class GAOptimizer(BaseOptimizerBase):
    """Genetic algorithm optimizer."""

    best_hps = config.known_hyperparameters["GAOptimizer"]["best"]

    def __init__(self, hp: Dict, solution_storage: Optional[Path] = None):
        """
        Initialize the GA optimizer with the given hyperparameters.

        :param hp: Dict of hyperparameters for pygad.GA; at least needs population_size.
        :param solution_storage: Where to store created solutions.
          If not provided, solutions are not stored or returned.
        """
        super().__init__(hp, solution_storage)
        self.hp = deepcopy(self.best_hps)
        self.hp.update(hp)

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        return {"alg": "GAOptimizer", "hp": self.hp}

    def _on_generation(self, _):
        """Extend best_solution_info and test timeout."""
        if process_time() - self._t0 > self._timeout:
            return "stop"

    def _optimize(self):
        """
        Optimize the base of the assembly in the given environment with a Genetic Algorithm.
        """
        local_hp = self.hp.copy()
        initial_population = [self._env.action_space.sample() for _ in range(local_hp.pop("population_size"))]
        gene_space = [{"low": low, "high": high}
                      for low, high in zip(self._env.action_space.low, self._env.action_space.high)]

        # Create GA
        ga_instance = pygad.GA(
            num_genes=len(initial_population[0]),
            fitness_func=lambda ga_instance, action, idx: self._evaluate(action),
            gene_space=gene_space,
            gene_type=float,
            initial_population=initial_population,
            on_generation=self._on_generation,
            **local_hp
        )

        # Run GA
        ga_instance.run()


class BOOptimizer(BaseOptimizerBase):
    """Bayesian optimization optimizer."""

    best_hps = config.known_hyperparameters["BOOptimizer"]["best"]

    def __init__(self, hp: Dict, solution_storage: Optional[Path] = None):
        """Set hyper-parameters for skopt.gp_minimize."""
        super().__init__(hp, solution_storage)
        self.hp = deepcopy(self.best_hps)
        self.hp.update(hp)

    @property
    def specs(self) -> Dict:
        """Return a dictionary with the specs of the optimizer."""
        return {
            "alg": "BOOptimizer",
            "hp": self.hp
        }

    def _evaluate(self, action: np.ndarray, opt_with_time: bool = False) -> Union[float, Tuple[float, float]]:
        """Evaluate the action in the environment. Optionally append calculation time as feedback to optimizer."""
        if not opt_with_time:
            return super()._evaluate(action)
        t0 = process_time()
        reward = super()._evaluate(action)
        return reward, process_time() - t0

    def _optimize(self):
        """Optimize with bayesian optimization."""
        hp = self.hp.copy()
        batch_size = hp.pop("batch_size")
        ask_strategy = hp.pop("ask_strategy")
        opt = skopt.Optimizer(tuple(zip(self._env.action_space.low, self._env.action_space.high)),
                              **hp)
        opt_with_time = "ps" in hp['acq_func']  # Whether to use time as input for acq_func
        while self.time_left:
            if batch_size > 1:
                actions = opt.ask(batch_size, strategy=ask_strategy)
                obs = tuple(self._evaluate(a, opt_with_time) for a in np.asarray(actions))
                opt.tell(actions, obs)
            else:
                action = opt.ask()
                opt.tell(action, self._evaluate(np.asarray(action), opt_with_time))


str_to_base_optimizer = {
    "RandomBaseOptimizer": RandomBaseOptimizer,
    "DummyOptimizer": DummyOptimizer,
    "RandomGrid": RandomGrid,
    "GAOptimizer": GAOptimizer,
    "BOOptimizer": BOOptimizer,
    "AdamOptimizer": AdamOptimizer
}
