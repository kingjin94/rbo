"""
Tests of the various gymnasium compliant environments formalizing the optimization problem.

:author: Matthias Mayer
:date: 15.12.23
"""
from copy import deepcopy
import gymnasium as gym
import time
import unittest
import numpy as np
import numpy.testing as np_test

import cobra.task.task

from timor import Transformation
from timor.configuration_search.AssemblyFilter import InverseKinematicsSolvable, RobotCreationFilter
from timor.task import CostFunctions, Task
from timor.task.Constraints import BasePlacement, CollisionFree
from timor.task.Goals import Pause
from timor.task.Tolerance import CartesianXYZ
from timor.utilities import spatial
from timor.utilities.tolerated_pose import ToleratedPose

from base_opt.utilities import Proxies
from base_opt.utilities.AsssemblyFilter import RobotLongEnoughFilter
from mcs.TaskSolver import SimpleHierarchicalTaskSolverWithoutBaseChange
from mcs.utilities.debug_solution import debug_solution
from mcs.utilities.default_robots import get_six_axis_modrob_v2


class BaseChangeEnvironmentTests(unittest.TestCase):
    """Test the base change environments."""

    def setUp(self):
        """Setup demo task, assembly, and task solver."""
        ptp_1_file = cobra.task.get_task(id='simple/PTP_1')
        # eval_data = get_tasks_solutions(re.compile('PTP_1'))
        self.task = Task.Task.from_json_file(ptp_1_file)
        self.multi_whitman_tasks = [cobra.task.get_task(id=f'Whitman2020/with_torque/3g_3o/{i}') for i in range(4)]
        self.task_whitman = Task.Task.from_json_file(self.multi_whitman_tasks[1])
        self.assembly = get_six_axis_modrob_v2()
        # Create solver that can handle this task (see test_task_solver.py)
        self.task_solver = self.task_solver_creator(self.task)

    @staticmethod
    def task_solver_creator(task, timeout=1.):
        """Create a simple hierarchical task solver that can solve the given task."""
        return SimpleHierarchicalTaskSolverWithoutBaseChange(
            task, timeout=timeout,
            filters=(RobotCreationFilter(), InverseKinematicsSolvable(max_iter=100)))

    def test_base_change_environment(self, n_trials=10):
        """
        Test the base change environment.
        """
        # Make sure errors are thrown if non CartesianXYZ, no base constraint
        env = Proxies.BaseChangeEnvironment(self.assembly, self.task_solver, CostFunctions.CycleTime(),
                                            observations=('goal_poses',))
        self.assertEqual(env.observation_space, gym.spaces.dict.Dict({}))  # No observation
        self.assertEqual(len(env.task.base_constraint.tolerance.tolerances), 0)  # Empty tolerance
        o, r, _, _, _ = env.step(env.action_space.sample())
        self.assertIsNotNone(o)
        self.assertGreaterEqual(r, env.reward_fail)
        env.task.constraints = [c for c in env.task.constraints if not isinstance(c, BasePlacement)]
        with self.assertRaises(ValueError):  # No has no base constraint
            env.step(env.action_space.sample())
        other_base_constraint = BasePlacement(ToleratedPose(
            Transformation.from_translation((1, 2, 3)), CartesianXYZ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))))
        env.task.constraints.append(other_base_constraint)
        # Check that action changes robot's base
        rot = env.assembly.robot.placement.rotation
        for _ in range(n_trials):
            action = env.action_space.sample()  # in [-1, 1]^3
            env.step(action)  # now should be possible with correct constraint
            np_test.assert_almost_equal(
                env.assembly.robot.placement.translation,
                action * .1 + other_base_constraint.base_pose.nominal.in_world_coordinates().translation)
            np_test.assert_almost_equal(env.assembly.robot.placement.rotation, rot)

        # Test error != At Goal
        task = deepcopy(self.task)
        task.goals = [*task.goals, Pause("tmp", 1.)]
        self.assertEqual(len(task.goals), 3)
        with self.assertRaises(ValueError):
            _ = Proxies.BaseChangeEnvironment(self.assembly, self.task_solver, CostFunctions.CycleTime(), task=task)

        # Test that observed goals and projected actions are relative to the base constraint
        # These are sets with nan in all remaining rows
        test_env = Proxies.BaseChangeEnvironment(self.assembly, self.task_solver, CostFunctions.CycleTime(),
                                                 observations=('goal_poses',))
        # Test that actions are projected to poses satisfying the base pose constraint -> action2pose
        self.assertTrue(test_env.task.base_constraint.base_pose.valid(test_env.action2base_pose(np.zeros(3))))
        self.assertTrue(test_env.task.base_constraint.base_pose.valid(test_env.action2base_pose(np.ones(3))))
        self.assertTrue(test_env.task.base_constraint.base_pose.valid(test_env.action2base_pose(-1. * np.ones(3))))
        self.assertFalse(test_env.task.base_constraint.base_pose.valid(test_env.action2base_pose(1.1 * np.ones(3))))

        # Test other action
        env = Proxies.BaseChangeEnvironment(self.assembly, self.task_solver, CostFunctions.CycleTime(),
                                            action2base_pose='xyz_rotvec')  # Default observation should work here
        self.assertTupleEqual(env.action_space.shape, (6,))
        self.assertEqual(
            env.action2base_pose(np.asarray((0., 0., 0., 1., 0., 0.))),
            env.task.base_constraint.base_pose.nominal.in_world_coordinates() @ spatial.rotX(np.pi)
        )
        self.assertEqual(
            env.action2base_pose(np.asarray((0., 0., 0., 0, -.5, 0.))),
            env.task.base_constraint.base_pose.nominal.in_world_coordinates() @ spatial.rotY(-np.pi / 2)
        )
        self.assertEqual(
            env.action2base_pose(np.asarray((0., 0., 0., 0., 0., .25))),
            env.task.base_constraint.base_pose.nominal.in_world_coordinates() @ spatial.rotZ(np.pi / 4)
        )


if __name__ == '__main__':
    unittest.main()
