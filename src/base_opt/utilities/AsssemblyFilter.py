from functools import lru_cache
from typing import Collection, Tuple, Type

import numpy as np

from timor import ModuleAssembly, PinRobot
from timor.configuration_search.AssemblyFilter import GOALS_WITH_ONE_POSE, GoalByGoalFilter, \
    IntermediateFilterResults, ResultType
from timor.task import Goals, Task
from timor.utilities.frames import Frame


class RobotLongEnoughFilter(GoalByGoalFilter):
    """
    Check if the robot is long enough to reach each goal in a task at the current base placement.

    Over-approximates the robot length by summing up the lengths of all links.
    """

    requires: Tuple[ResultType] = (ResultType.robot,)
    provides: Tuple[ResultType] = ()
    goal_types_filtered: Collection[Type[Goals.GoalBase]] = GOALS_WITH_ONE_POSE

    @staticmethod
    @lru_cache
    def _over_approximate_robot_length(robot: PinRobot) -> float:
        """Calculate an over-approximation of a given robot's reach by summing translations between joints."""
        if not isinstance(robot, PinRobot):
            raise NotImplementedError("Can only calculate length for PinRobot")

        # Iterate backwards from TCP to base, summing up the lengths of all links
        current_frame = robot.tcp
        # First offset static on last link via frame placement
        lengths = [np.linalg.norm(robot.model.frames[current_frame].placement.translation)]
        current_frame = robot.model.frames[current_frame].parent
        # Other offsets via joint placements
        while current_frame > 0:
            delta_J = robot.model.jointPlacements[current_frame].translation.copy()  # Copy to avoid side effects
            current_frame = robot.model.parents[current_frame]
            if current_frame == 0:  # Placement of robot rolled into first transformation
                delta_J -= robot.placement.translation
            lengths.append(np.linalg.norm(delta_J))
        return sum(lengths)

    def _check_given(self, assembly: ModuleAssembly, goal: Goals.GoalBase, results: IntermediateFilterResults,
                     task: Task):
        return self._check_goal(assembly, goal, results, task)  # Rather fast anyway

    def _check_goal(self, assembly: ModuleAssembly, goal: Goals.GoalBase, results: IntermediateFilterResults,
                    task: Task):
        """Make sure robot length >= distance goal <-> robot placement"""
        goal_pose = goal.goal_pose.nominal
        if isinstance(goal_pose, Frame):
            goal_pose = goal_pose.in_world_coordinates()
        return (self._over_approximate_robot_length(results.robot)
                >= np.linalg.norm((results.robot.placement.inv @ goal_pose).translation))

    def __repr__(self):
        """Debug string to identify this filter"""
        return f"{self.__class__.__name__}"