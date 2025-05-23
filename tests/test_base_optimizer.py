"""
Tests of the various optimizers implemented in base_opt.

:author: Matthias Mayer
:date: 15.12.23
"""
from collections import namedtuple
import itertools
from pathlib import Path
import tempfile
from time import process_time, time
import unittest

import numpy as np
import numpy.testing as np_test
import optuna
import pandas as pd
from timor.utilities.spatial import rotX
import torch

import cobra.task.task

from mcs.PathPlanner import PathPlanningFailedException
from mcs.TaskSolver import SimpleHierarchicalTaskSolverWithoutBaseChange
from mcs.utilities.default_robots import get_six_axis_modrob_v2
from mcs.utilities.trajectory import TrajectoryGenerationError

from timor import Transformation
from timor.configuration_search.AssemblyFilter import default_filters
from timor.task import CostFunctions, Tolerance
from timor.task.Constraints import BasePlacement
from timor.task.Solution import SolutionHeader, SolutionTrajectory
from timor.task.Task import Task
from timor.utilities.tolerated_pose import ToleratedPose

from base_opt.base_opt import config, evaluate_algorithm, hyperparameter_optimization
from base_opt.base_opt.BaseOptimizer import AdamOptimizer, BOOptimizer, BaseOptimizationHistory, BaseOptimizerBase, \
    DummyOptimizer, GAOptimizer, RandomBaseOptimizer, RandomGrid
from base_opt.utilities.Proxies import BaseChangeEnvironment


class TestBaseOptimizationInfo(unittest.TestCase):
    """Test the BaseOptimizationInfo and BaseOptimizationHistory classes."""

    def test_add_step(self, reward_fail=-234.56, reward_better=-20, run_time=0.5):
        """Test that steps are added correctly to the history."""
        hist = BaseOptimizationHistory(reward_fail)
        fake_solution = namedtuple('SolutionTrajectory', ('valid', ))(True)
        hist.add_step(np.zeros((6,)), reward_better, {'solution': fake_solution, 'is_success': True}, run_time)
        self.assertTrue(hist.any_valid)
        self.assertEqual(hist.first_success_idx, 0)
        self.assertAlmostEqual(hist.first_success_time, run_time)
        self.assertEqual(hist.best_solution, None)  # No storage location -> no UUID
        self.assertEqual(hist.best_reward, reward_better)
        self.assertEqual(hist.best_valid_reward, reward_better)
        np_test.assert_equal(hist.best_action, np.zeros((6,)))
        self.assertEqual(len(hist), 1)
        self.assertEqual(len(hist.to_data_frame()), 1)

    def test_add_step_no_solution(self, reward_fail=-234.56, reward_better=-20, run_time=0.5):
        """Test that steps are added correctly to the history."""
        hist = BaseOptimizationHistory(reward_fail)
        hist.add_step(np.zeros((6,)), reward_better, {'is_success': False}, run_time)
        self.assertFalse(hist.any_valid)
        self.assertIsNone(hist.first_success_idx)
        self.assertIsNone(hist.first_success_time)
        self.assertIsNone(hist.best_solution)
        self.assertEqual(hist.best_reward, reward_better)
        self.assertEqual(hist.best_valid_reward, reward_fail)
        np_test.assert_equal(hist.best_action, None)
        self.assertEqual(len(hist), 1)
        self.assertEqual(len(hist.to_data_frame()), 1)
        hist.add_step(np.zeros((6,)), reward_better, {'is_success': True}, run_time)
        self.assertTrue(hist.any_valid)
        self.assertEqual(hist.best_valid_reward, reward_better)
        self.assertEqual(len(hist), 2)

    def test_handling_empty_history(self, reward_fail=-234.56):
        """Test that empty history is handled correctly."""
        hist = BaseOptimizationHistory(reward_fail)
        self.assertFalse(hist.any_valid)
        self.assertIsNone(hist.first_success_idx)
        self.assertIsNone(hist.first_success_time)
        self.assertIsNone(hist.best_solution)
        self.assertEqual(hist.best_reward, reward_fail)
        self.assertEqual(hist.best_valid_reward, reward_fail)
        self.assertIsNone(hist.best_action)
        self.assertEqual(hist.reward_fail, reward_fail)
        self.assertEqual(len(hist), 0)
        self.assertEqual(len(hist.to_data_frame()), 0)


class TestBaseOptimizer(unittest.TestCase):
    """Test that base pose optimizer work in principle on simple environment and task."""

    def setUp(self):
        """Set up the test case; load assembly, example tasks, base change environment"""
        self.assembly = get_six_axis_modrob_v2()
        self.task = Task.from_json_file(cobra.task.get_task(id='simple/PTP_1'))
        new_constraints = [c for c in self.task.constraints if not isinstance(c, BasePlacement)]
        new_constraints.append(BasePlacement(ToleratedPose(
            self.task.base_constraint.base_pose.nominal,
            Tolerance.CartesianXYZ((-.1, .1), (-.1, .1), (-.1, .1))  # Not too far away from the original base
        )))
        self.task.constraints = new_constraints
        self.task_solver = SimpleHierarchicalTaskSolverWithoutBaseChange(
            self.task, assembly=self.assembly, filters=default_filters(self.task), timeout=1.)
        self.single_step_env = BaseChangeEnvironment(self.assembly, self.task_solver,
                                                     CostFunctions.CycleTime(),
                                                     self.task, reward_fail=-20.)
        self.single_step_env_rot = BaseChangeEnvironment(self.assembly, self.task_solver,
                                                         CostFunctions.CycleTime(),
                                                         self.task, reward_fail=-20.,
                                                         action2base_pose='xyz_rotvec')
        self.timeout_base_opt = 20.  # Timeout for base optimizer in seconds

    def _assert_success(self, action, reward, info):
        """Helper that makes sure that the optimizer found a valid solution and checks that returned info consistent."""
        self.assertIsNotNone(action)  # This task is really simple and should be always solvable
        self.assertIsNotNone(info)
        self.assertIsNotNone(info.best_solution)
        self.assertTrue(info.is_success)
        self.assertTrue(info.best_solution.valid)
        self.assertEqual(reward, -info.best_solution.cost)
        self.assertIsInstance(info.best_base_pose, Transformation)
        self.assertTrue(np.all(self.single_step_env.action_space.low[None, :] <= info.history.actions),
                        f"Action history: {info.history.actions} outside limits: {self.single_step_env.action_space}")
        self.assertTrue(np.all(info.history.actions <= self.single_step_env.action_space.high[None, :]),
                        f"Action history: {info.history.actions} outside limits: {self.single_step_env.action_space}")
        self.assertEqual(info.task_id, self.task.id)

    def _test_optimizer(self, optimizer_class, hps=None):
        """Helper that tests an optimizer on the given environment and task; ensures timeout is kept"""
        with tempfile.TemporaryDirectory() as d:
            optimizer = optimizer_class({} if hps is None else hps, solution_storage=d)
            t0 = process_time()
            best_action, best_reward, info = optimizer.optimize(self.single_step_env, timeout=self.timeout_base_opt)
            self.assertLess(process_time(), t0 + self.timeout_base_opt * 1.1 + 1.)  # Not way longer
            self._assert_success(best_action, best_reward, info)
            self.assertIsInstance(BaseOptimizerBase.from_specs(optimizer.specs),
                                  optimizer_class)  # Test (de)serialization
            for sol_id in info.history.solutions:
                if sol_id is not None:
                    self.assertTrue(Path(d + f'/solution-{sol_id}.json').is_file(),
                                    f"Solution {sol_id} not stored in {d}")
            return best_action, best_reward, info

    def test_random_base_optimizer(self):
        """Test that the random base optimizer works on a simple task and environment."""
        self._test_optimizer(RandomBaseOptimizer)

    def test_dummy_optimizer(self):
        """Test that the dummy optimizer works on a simple task and environment."""
        self._test_optimizer(DummyOptimizer)

    def test_greedy_optimizer(self):
        """Test that the greedy optimizer works on a simple task and environment."""
        self._test_optimizer(AdamOptimizer)

    def test_adam_optimizer_improves_action(self, num_pts=8):
        """Test whether greed optimizer improves a really poor action in big search space."""
        self.task.base_constraint.tolerance._a *= 10
        self.task.base_constraint.tolerance._b *= 10
        self.task.base_constraint.tolerance._c *= 10
        opt = AdamOptimizer(dict(local_search_steps=10, lr=.2, local_ik_steps=100))
        result_new = []
        result_old = []
        optimized_actions = []
        local_opt_time = 0
        for i in range(num_pts):
            action = np.array((i % 2, (i // 2) % 2, i // 4)) * 2. - 1.  # Edge of action space
            opt._reset_next_action(self.single_step_env)
            t0 = process_time()
            new_action = opt._improve_guess(self.single_step_env, action)
            optimized_actions.append(new_action)
            local_opt_time += process_time() - t0
            # self.assertTrue(np.any(new_action != action))
            result_old += [self.single_step_env.step(action) for _ in range(10)]
            self.assertTrue(self.assembly.robot.placement == self.single_step_env.action2base_pose(action))
            result_new += [self.single_step_env.step(new_action) for _ in range(10)]
            self.assertTrue(self.assembly.robot.placement == self.single_step_env.action2base_pose(new_action))
        self.assertGreater(np.mean([r[4].get('filter_fail', False) for r in result_old]),
                           np.mean([r[4].get('filter_fail', False) for r in result_new]),
                           "Greedy optimizer did not improve number of filter fails")
        self.assertGreater(np.mean([r[1] for r in result_new]),
                           np.mean([r[1] for r in result_old]),
                           "Greedy optimizer did not improve reward")
        print(f"Local optimization took {local_opt_time / num_pts} seconds per try.")
        print("Optimized actions mean+/-std: "
              f"{np.mean(optimized_actions, axis=0)} +/- {np.std(optimized_actions, axis=0)}")
        print(f"Optimized actions: {optimized_actions}")
        self.task.base_constraint.tolerance._a /= 10
        self.task.base_constraint.tolerance._b /= 10
        self.task.base_constraint.tolerance._c /= 10

    def test_adam_optimizer_improves_action_rotvec(self, num_pts=8):
        """test adam also works in 6D xyz + rotvec space"""
        self.task.base_constraint.tolerance._a *= 10
        self.task.base_constraint.tolerance._b *= 10
        self.task.base_constraint.tolerance._c *= 10
        opt = AdamOptimizer(dict(local_search_steps=10, lr=.2, local_ik_steps=100))
        result_new = []
        result_old = []
        optimized_actions = []
        local_opt_time = 0
        for i in range(num_pts):
            action = np.array((i % 2, (i // 2) % 2, i // 4, i % 2, (i // 2) % 2, i // 4)) * 2. - 1.
            opt._reset_next_action(self.single_step_env_rot)
            t0 = process_time()
            new_action = opt._improve_guess(self.single_step_env_rot, action)
            optimized_actions.append(new_action)
            local_opt_time += process_time() - t0
            result_old += [self.single_step_env_rot.step(action) for _ in range(10)]
            self.assertTrue(self.assembly.robot.placement == self.single_step_env_rot.action2base_pose(action))
            result_new += [self.single_step_env_rot.step(new_action) for _ in range(10)]
            self.assertTrue(self.assembly.robot.placement == self.single_step_env_rot.action2base_pose(new_action))
        self.assertGreater(np.mean([r[4].get('filter_fail', False) for r in result_old]),
                           np.mean([r[4].get('filter_fail', False) for r in result_new]),
                           "Greedy optimizer did not improve number of filter fails")
        self.assertGreater(np.mean([r[1] for r in result_new]),
                           np.mean([r[1] for r in result_old]),
                           "Greedy optimizer did not improve reward")
        print(f"Local optimization took {local_opt_time / num_pts} seconds per try.")
        print("Optimized actions mean+/-std: "
              f"{np.mean(optimized_actions, axis=0)} +/- {np.std(optimized_actions, axis=0)}")
        print(f"Optimized actions: {optimized_actions}")
        self.task.base_constraint.tolerance._a /= 10
        self.task.base_constraint.tolerance._b /= 10
        self.task.base_constraint.tolerance._c /= 10

    def test_grid_optimizer(self):
        """Test that the grid optimizer works and keeps to grid on a simple task and environment."""
        _, _, info = self._test_optimizer(RandomGrid, {'num_steps': 5})
        self.assertLessEqual(len(info.history.actions), 125)  # 5^3 actions
        self.assertEqual(len(info.history.actions), len(info.history.actions))

    def test_GA_optimizer(self):
        """Test that the GA optimizer works on a simple task and environment."""
        self._test_optimizer(GAOptimizer)

    def test_BO_optimizer(self):
        """Test that the BO optimizer works on a simple task and environment."""
        self._test_optimizer(BOOptimizer)

    def _test_eval_alg(self, alg_name):
        """Helper to test that an algorithm can be evaluated."""
        with tempfile.TemporaryDirectory() as d:
            raw_data_file = Path(d).joinpath('raw_data.csv')
            evaluate_algorithm.main(
                ['--eval-set', 'eval_min',  # Default timeout should fit this set
                 '--algorithm', alg_name,
                 '--solution-storage', d,
                 '--raw-data', str(raw_data_file)])

            # Test raw data is stored
            self.assertTrue(raw_data_file.is_file())
            raw_data = pd.read_csv(raw_data_file, keep_default_na=False)
            # Check data frame
            self.assertTrue(alg_name == raw_data['Algorithm'].unique(),
                            f"Wrong algorithm name in {raw_data_file}: {raw_data['Algorithm'].unique()}")
            for _, res_task in raw_data.groupby(['Task ID', 'Seed']):
                # A task result should be either all success or all fail
                self.assertEqual(len(res_task['Success'].unique()), 1)
                # Make sure if success any reward is better than fail; otherwise all smaller equal fail
                if res_task['Success'].any():
                    self.assertGreater(res_task['Reward'].max(), res_task['Reward Fail'].max())
                else:
                    self.assertGreaterEqual(res_task['Reward Fail'].min(), res_task['Reward'].max())

            # Check solutions exist if valid
            for sol_id in raw_data['Solution']:
                if sol_id != "":
                    self.assertTrue(Path(d + f'/solution-{sol_id}.json').is_file(),
                                    f"Solution {sol_id} not stored in {d}")

            for _, valid_step in raw_data.loc[raw_data['Solution'] != ""].iterrows():
                print(f"Checking solution {valid_step['Solution']}")
                self.assertTrue(Path(d + f'/solution-{valid_step["Solution"]}.json').is_file(),
                                f"Solution {valid_step} not stored in {d}")
                sol = SolutionTrajectory.from_json_file(
                    Path(d).joinpath(f'solution-{valid_step["Solution"]}.json'),
                    {t.id: t for t in config.str_to_task_set['eval_min'].as_finite_iterable()})
                self.assertEqual(sol.valid, valid_step['Valid Solution'])
                if sol.valid:
                    self.assertAlmostEqual(sol.cost, -valid_step['Reward'])
                else:
                    self.assertAlmostEqual(valid_step['Reward'], valid_step['Reward Fail'])

    def test_evaluate_algorithm_GA(self):
        """Test that GAOptimizer can be evaluated."""
        self._test_eval_alg('GAOptimizer')

    def test_evaluate_algorithm_BO(self):
        """Test that BOOptimizer can be evaluated."""
        self._test_eval_alg('BOOptimizer')

    def test_evaluate_algorithm_Adam(self):
        """Test that BOOptimizer can be evaluated."""
        self._test_eval_alg('AdamOptimizer')

    def _test_hyper_opt_alg(self, alg_name, additional_args=None, n_trials: int = 4):
        with tempfile.TemporaryDirectory() as d:
            storage = f'sqlite:///{d}/optuna.db'
            hyperparameter_optimization.main(
                ['--eval-set', 'eval_min',
                 '--timeout', '2',  # Quicker
                 '--task-solver-timeout', '1',  # Quicker
                 '--n-trials', str(n_trials),
                 '--parallel-trials', '2',  # Check multiprocessing
                 '--storage', storage,
                 '--algorithm', alg_name,
                 '--study-name', 'test',
                 '-optimize'
                 ] + (additional_args if additional_args is not None else []))
            # Test that all trials are evaluated
            self.assertTrue(Path(d + '/optuna.db').is_file())
            db = optuna.create_study(storage=storage, study_name='test', load_if_exists=True)
            self.assertGreaterEqual(len(db.trials), n_trials)
            self.assertNotEqual(db.trials[0].params, db.trials[1].params)  # Ensure different params tested
            self.assertIsNotNone(db.best_params)
            for t in db.trials:  # Check that each optimization run logged
                self.assertTrue(config.TMP_DIR.joinpath(t.user_attrs['detailed_results']).is_file())

    def test_hyper_opt_algorithm_GA(self):
        """Test that GAOptimizer can be hyperparameter optimized."""
        self._test_hyper_opt_alg('GAOptimizer')

    def test_hyper_opt_algorithm_BO(self):
        """Test that BOOptimizer can be hyperparameter optimized."""
        self._test_hyper_opt_alg('BOOptimizer')

    def test_hyper_opt_algorithm_Adam(self):
        """Test that BOOptimizer can be hyperparameter optimized."""
        self._test_hyper_opt_alg('AdamOptimizer')

    def test_success_chance_default_task_solver(self, test_n_tasks: int = 10):
        """Test how likely it is to solve each task with the default task solver at the nominal position."""
        import ompl.util  # Make sure that OMPL loadable and not too verbose
        ompl.util.setLogLevel(ompl.util.LOG_WARN)
        # for each task set
        for task_gen_name, task_gen in config.str_to_task_set.items():
            print(f"Testing success rate for {task_gen_name}")
            count = 0
            fails = 0
            invalid_sol = 0
            filter_fails = 0
            ik_solvable_but_filter_fail = 0
            solver_timeout = 0
            path_planning_failed = 0
            trajectory_gen_failed = 0
            for task in itertools.islice(task_gen, test_n_tasks):
                count += 1
                self.assembly.robot.set_base_placement(task.base_constraint.base_pose.nominal)
                solver = config.get_task_solver(task)
                solver.timeout = 10.
                traj = None
                t0 = time()
                try:
                    traj = solver.solve(self.assembly)
                except TimeoutError:
                    solver_timeout += 1
                except PathPlanningFailedException:
                    path_planning_failed += 1
                except TrajectoryGenerationError:
                    trajectory_gen_failed += 1
                solve_time = time() - t0
                fails += traj is None
                if traj is not None:
                    sol = SolutionTrajectory(
                        traj, SolutionHeader(task.id), task, self.assembly, CostFunctions.CycleTime(),
                        [self.assembly.robot.placement, ]
                    )
                    invalid_sol += not sol.valid
                filter_fail = {f: f.failed for f in solver.filters}
                if sum(filter_fail.values()) > 0:
                    print(f"Failed filters: {[f for f, count in filter_fail.items() if count > 0]}")
                filter_fails += sum(filter_fail.values()) > 0
                ik_solvable_default = [self.assembly.robot.ik(g.goal_pose, task=task)[1] for g in task.goals]
                ik_solvable_but_filter_fail += all(ik_solvable_default) and sum(filter_fail.values()) > 0
                # Cleanup
                [f.reset() for f in solver.filters]
                print(f"Trying to solve task {task.id} took {solve_time} seconds.")
            print(f"Planned for {count} tasks in {task_gen_name}")
            print(f"Fail rate: {100 * (fails / count)}%")
            print(f"Invalid sol rate: {100 * (invalid_sol / count)}%")
            print(f"Filter fail rate: {100 * (filter_fails / count)}%")
            print(f"IK solvable but filter fail rate: {100 * (ik_solvable_but_filter_fail / count)}%")
            print(f"Solver errors - Timeout: {solver_timeout}, Path planning failed: {path_planning_failed}, "
                  f"Trajectory generation failed: {trajectory_gen_failed}, Filter: {filter_fails}")
            print("========================================================")
            # TODO Any reasonable warning threshold for fail rate or similar?

    def test_success_rate_random(self):
        """Test success rate for each of the task sets with random base optimizer."""
        optimizer = RandomBaseOptimizer({})
        base_opt_timeout = 10.
        import ompl.util  # Make sure that OMPL loadable and not too verbose
        ompl.util.setLogLevel(ompl.util.LOG_WARN)
        for task_gen_name, task_gen in config.str_to_task_set.items():
            print(f"Testing success rate for {task_gen_name}")
            count = 0
            fails = 0
            rewards = []
            runtimes = []
            reward_history = []
            for task in itertools.islice(task_gen, 10):
                count += 1
                t0 = time()
                task_solver = config.get_task_solver(task)
                single_step_env = BaseChangeEnvironment(self.assembly, task_solver,
                                                        CostFunctions.CycleTime(),
                                                        task, reward_fail=-20.)
                best_action, best_reward, info = optimizer.optimize(single_step_env, timeout=base_opt_timeout)
                runtimes.append(time() - t0)
                fails += not info.is_success
                reward_history.append(info.history.rewards)
                if info.is_success:
                    rewards.append(best_reward)

            print(f"Planned for {count} tasks in {task_gen_name}")
            print(f"Fail rate: {100 * (fails / count)}%")
            print(f"Average reward on success: {np.mean(rewards)} +/- {np.std(rewards)}")
            print(f"Average runtime: {np.mean(runtimes)} +/- {np.std(runtimes)}")
            print(f"Average number of tested actions: {np.mean([len(r) for r in reward_history])}")

    def test_adam_pytorch_gradients(self, n_trials=100):
        """Test that PyTorch gradients are correctly computed."""
        opt = AdamOptimizer({})
        for _ in range(n_trials):
            a = self.single_step_env_rot.action_space.sample()
            a_tensor = torch.tensor(a, requires_grad=True)
            T_base = self.single_step_env_rot.action2base_pose(a_tensor)
            self.assembly.robot.set_base_placement(T_base.detach().numpy())
            q_sol = {g_id: self.assembly.robot.ik(g.goal_pose) for g_id, g in self.task.goals_by_id.items()}
            opt._reset_next_action(self.single_step_env_rot)
            cost = torch.zeros(1)
            for g_id, (q, success) in q_sol.items():
                T_rel_base = self.assembly.robot.placement.inv @ self.assembly.robot.fk(q)
                T_des = self.single_step_env_rot.task.goals_by_id['1'].goal_pose.nominal.in_world_coordinates()
                cost += opt._tensor_ik_cost_function(
                    a_tensor, torch.tensor(T_rel_base.homogeneous, dtype=a_tensor.dtype),
                    torch.tensor(T_des.homogeneous, dtype=a_tensor.dtype))
            cost.backward()
            self.assertIsNotNone(a_tensor.grad)

    def test_adam_pytorch_gradients_corner_case(self):
        """Test that PyTorch gradients are correctly computed."""
        opt = AdamOptimizer({})
        opt._reset_next_action(self.single_step_env_rot)
        # Corner cases
        a_tensor = torch.tensor((0.5, 0.5, 0.5, 0.01, 0.01, 0.01), requires_grad=True)
        T_base = self.single_step_env_rot.action2base_pose(a_tensor)
        self.assembly.robot.set_base_placement(T_base.detach().numpy())
        q_sol = {g_id: self.assembly.robot.ik(g.goal_pose) for g_id, g in self.task.goals_by_id.items()}
        self.assertTrue(all(q[1] for q in q_sol.values()))
        T_sol = {g_id: self.assembly.robot.placement.inv @ self.assembly.robot.fk(q[0]) for g_id, q in q_sol.items()}
        for g_id, T in T_sol.items():
            T_des = self.single_step_env_rot.task.goals_by_id['1'].goal_pose.nominal.in_world_coordinates()
            # All the same
            opt._tensor_ik_cost_function(a_tensor,
                                         torch.tensor(T.homogeneous, dtype=a_tensor.dtype),
                                         torch.tensor(T_des.homogeneous, dtype=a_tensor.dtype)).backward()
            self.assertIsNotNone(a_tensor.grad)
            self.assertLess(a_tensor.grad[:3].norm(), 0.2)  # Close by in translation even with IK tolerance
            a_tensor.grad = None
            # 180-degree rotation
            T_rotated = T.homogeneous.copy()
            T_rotated[:3, :3] = rotX(np.pi)[:3, :3] @ T_rotated[:3, :3]
            opt._tensor_ik_cost_function(a_tensor,
                                         torch.tensor(T_rotated, dtype=a_tensor.dtype),
                                         torch.tensor(T_des.homogeneous, dtype=a_tensor.dtype)).backward()
            self.assertIsNotNone(a_tensor.grad)
            self.assertLess(a_tensor.grad[:3].norm(), 0.2)  # Translation close by even with IK tolerance
            self.assertLess(.1, a_tensor.grad[3:].norm())  # Rotation with long arm more effective
            a_tensor.grad = None
            # Almost 180 degrees
            T_rotated = T.homogeneous.copy()
            T_rotated[:3, :3] = rotX(0.99 * np.pi)[:3, :3] @ T_rotated[:3, :3]
            opt._tensor_ik_cost_function(a_tensor,
                                         torch.tensor(T_rotated, dtype=a_tensor.dtype),
                                         torch.tensor(T_des.homogeneous, dtype=a_tensor.dtype)).backward()
            self.assertIsNotNone(a_tensor.grad)
            self.assertLess(a_tensor.grad[:3].norm(), 0.2)  # Translation close by even with IK tolerance
            self.assertLess(.1, a_tensor.grad[3:].norm())  # Rotation with long arm more effective
            a_tensor.grad = None
