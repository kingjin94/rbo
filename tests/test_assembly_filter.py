import unittest

from timor import Transformation
from timor.configuration_search.AssemblyFilter import IntermediateFilterResults
from timor.task.Goals import At
from timor.task.Task import Task, TaskHeader
from timor.utilities.prebuilt_robots import get_six_axis_assembly
from timor.utilities.spatial import random_rotation
from timor.utilities.tolerated_pose import ToleratedPose

from base_opt.utilities.AsssemblyFilter import RobotLongEnoughFilter


class TestAssemblyFilter(unittest.TestCase):
    """Test MCS's assembly filter."""

    def setUp(self):
        """Create assembly for testing."""
        self.assembly = get_six_axis_assembly()
        self.assembly_backup = get_six_axis_assembly()

    def test_robot_length_filter(self):
        """Test that the robot length filter works."""
        intermediate_result = IntermediateFilterResults(robot=self.assembly.robot)
        filter = RobotLongEnoughFilter()
        max_reach = filter._over_approximate_robot_length(self.assembly.robot)
        max_dist_evidence = 0
        for _ in range(1000):
            dist = self.assembly.robot.fk(self.assembly.robot.random_configuration()).norm.translation_euclidean
            self.assertLess(dist, max_reach,
                            f"Any configuration should have less reach; not so {self.assembly.robot.configuration}")
            max_dist_evidence = max(max_dist_evidence, dist)
        print("Max reach via sampling: ", max_dist_evidence, "; Max reach: ", max_reach)
        for g_translation in ((1., 0, 0),
                              (1., 1., 1.),
                              (0, 0, 0),
                              (-1, 0, 0),
                              (-1, -1, -1)):  # Known valid
            task = Task(TaskHeader('test_task'),
                        goals=[At('test', ToleratedPose(Transformation.from_translation(g_translation))), ])
            self.assertTrue(filter.check(self.assembly, task, intermediate_result))

        for _ in range(1000):
            offset_base = Transformation.random()
            self.assembly.robot.set_base_placement(offset_base)
            direction = offset_base @ Transformation.from_rotation(random_rotation())

            filter._over_approximate_robot_length.cache_clear()  # Make sure _check_given executed
            task = Task(TaskHeader('test_task'),
                        goals=[At('test', ToleratedPose(
                            direction @ Transformation.from_translation((max_dist_evidence, 0, 0)))), ])
            self.assertTrue(filter.check(self.assembly, task, intermediate_result))
            task = Task(TaskHeader('test_task'),
                        goals=[At('test', ToleratedPose(
                            direction @ Transformation.from_translation((max_reach * 1.01, 0, 0)))), ])
            self.assertFalse(filter.check(self.assembly, task, intermediate_result))
            # Make sure filter does not change assembly's robot kinematics
            q = self.assembly.robot.random_configuration()
            self.assertTrue(
                ToleratedPose(self.assembly.robot.fk(q)).valid(
                    offset_base @ self.assembly_backup.robot.fk(q)
                )
            )