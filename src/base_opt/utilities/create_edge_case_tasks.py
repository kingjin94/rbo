from datetime import datetime
import itertools
from typing import Optional, Sequence, Type, Union

import numpy as np
from timor.Geometry import Box
from timor.task.Task import Task
from timor.task.Constraints import BasePlacement, SelfCollisionFree
from timor.task.Goals import Reach
from timor.task.Obstacle import Obstacle
from timor.task.Task import TaskHeader
from timor.task.Tolerance import CartesianXYZ
from timor.utilities.tolerated_pose import ToleratedPose
from timor.utilities.transformation import Transformation, TransformationLike

from base_opt.utilities.file_locations import DATA
from mcs.optimize.Sampler import CartesianGridSampler, GridIndexSampler, UniformRotationSampler
from mcs.optimize.TaskGenerator import TaskGenerator

"""
Create tasks at the edge of solvability by hiding readch goals inside obstacles and placing obstacles at the edge of the
base-placement domain.

:author: Friedrich Dang, Matthias Mayer
"""


R1 = np.array([[0, 0, 1],
              [0, 1, 0],
              [-1, 0, 0]])

R2 = np.array([[0, 0, 1],
              [-1, 0, 0],
              [0, -1, 0]])
R3 = np.array([[-0.17470792, -0.31766705, -0.93196823],
               [0.05754346, 0.94161355, -0.33174188],
               [0.98293738, -0.11158661, -0.14622769]])


HARD_TASKS = {  # Shape definition; goal voxel idx, obstacle voxel idx, rotation matrix for goal orientation
    "middle_oven": [((3, -3, 0), (-1, 2, 0), (-4, 4, 2)),   # seems to be easier
                    ((-3, 4, 2), (-4, 4, 1), (-4, 4, 3),
                     (-3, 4, 3), (-3, 4, 1),
                     (-4, 3, 2), (-4, 3, 3), (-4, 3, 1)),
                    R1],
    "oven": [((3, -3, 0), (-1, 3, 0), (-4, 4, 1)),   # seems to be easier
             ((-3, 4, 2), (-4, 4, 2), (-4, 4, 3),
              (-3, 4, 3), (-3, 4, 1),
              (-4, 3, 2), (-4, 3, 3), (-4, 3, 1)),
             R2],
    "cross": [((1, -1, 0), (-1, 2, 0), (-4, 4, 2)),  # seems to be easier
              ((-3, 4, 2),
               (-4, 4, 1), (-4, 4, 3),
               (-4, 3, 2)),
              R2],
    "mystic_cube": [((1, 4, 2), (1, 3, 1), (-4, 4, 2)),
                    ((-4, 4, 1), (-4, 4, 3),
                     (-3, 3, 3), (-3, 3, 1),
                     (-3, 5, 3), (-3, 5, 1),
                     (-4, 3, 2),
                     (-5, 5, 1), (-5, 5, 3),
                     (-5, 3, 1), (-5, 3, 3)),
                    R2],
    "equality_sim": [((1, -1, 0), (-1, 2, 0), (-4, 4, 2)),   # seems to be easier
                     ((-4, 4, 1), (-4, 4, 3),
                      (-3, 4, 3), (-3, 4, 1),
                      (-3, 3, 3), (-3, 3, 1),
                      (-3, 5, 3), (-3, 5, 1),
                      (-4, 3, 3), (-4, 3, 1),
                      (-4, 5, 1), (-4, 5, 3),
                      (-5, 5, 1), (-5, 5, 3),
                      (-5, 4, 1), (-5, 4, 3),
                      (-5, 3, 1), (-5, 3, 3)),
                     R2],
    "cross_cube": [((1, -1, 0), (-1, 2, 0), (-4, 4, 2)),
                   ((-3, 4, 3), (-3, 4, 1),
                    (-3, 3, 2),
                    (-4, 3, 3), (-4, 3, 1),
                    (-4, 5, 1), (-4, 5, 3),
                    (-5, 5, 2),
                    (-5, 4, 1), (-5, 4, 3),
                    (-5, 3, 2)),
                   R3],
    "house": [((0, 6, 2), (-2, 5, 3), (-4, 4, 2)),
              ((-4, 4, 1), (-4, 4, 3),
               (-3, 4, 3), (-3, 4, 1),
               (-3, 3, 1), (-3, 3, 2),
               (-3, 5, 3), (-3, 5, 1),
               (-4, 3, 3), (-4, 3, 1),
               (-4, 5, 1), (-4, 5, 3),
               (-5, 5, 1), (-5, 5, 3), (-5, 5, 2),
               (-5, 4, 1), (-5, 4, 3),
               (-5, 3, 1), (-5, 3, 3), (-5, 3, 2)),
              R3],
    "c_sim": [((0, 5, 1), (1, 4, 1), (-4, 4, 2)),
              ((-4, 4, 1), (-4, 4, 3),
               (-3, 4, 3), (-3, 4, 1), (-3, 4, 2),
               (-3, 3, 1), (-3, 3, 2),
               (-3, 5, 2), (-3, 5, 3), (-3, 5, 1),
               (-4, 3, 3), (-4, 3, 1),
               (-4, 5, 1), (-4, 5, 3),
               (-5, 5, 1), (-5, 5, 3),
               (-5, 4, 1), (-5, 4, 3),
               (-5, 3, 1), (-5, 3, 3)),
              R2],
    "XI_cube": [((1, -1, 0), (-2, 2, 0), (-4, 4, 2)),
                ((-4, 4, 1), (-4, 4, 3),
                 (-3, 4, 1),
                 (-3, 3, 3), (-3, 3, 1),
                 (-3, 5, 3), (-3, 5, 1),
                 (-4, 3, 2),
                 (-5, 5, 1), (-5, 5, 3),
                 (-5, 4, 1),
                 (-5, 3, 1), (-5, 3, 3)),
                R3],
    "mystic_field": [((1, -1, 0), (-1, 2, 0), (-4, 4, 2)),   # seems to be easier
                     ((-3, 4, 1), (-3, 4, 3),
                      (-2, 3, 3), (-1, 3, 1),
                      (-2, 5, 3), (-1, 5, 1),
                      (-4, 2, 2), (-4, 6, 2),
                      (-5, 5, 1), (-4, 5, 3),
                      (-5, 3, 1), (-4, 3, 3)),
                     R2]
}


class HardTaskGenerator(TaskGenerator):
    """
    Task generator for these harder tasks.
    """

    def __init__(self,
                 grid_size: Union[int, Sequence[int]],
                 spacing: Union[float, Sequence[float]],
                 offset: TransformationLike = Transformation.neutral(),
                 goal_orientation_sampler: Optional[Union[UniformRotationSampler, Type[UniformRotationSampler]]] = UniformRotationSampler,  # noqa: E501
                 id_prefix: str = 'task-',
                 rng: np.random.Generator = None,
                 base_tasks=list(HARD_TASKS.keys()),
                 diff_level="medium"
                 ):
        """
        Task generator for marginal tasks, esp., by hiding reach goals in between obstacles and moving the goals to the

        edge of reachable space is robot is placed optimally.

        :param grid_size: Task is created with a voxel grid with this number of voxels in each dimension (suggestion 5).
        :param spacing: The size of each voxel; for the suggested 5 voxels .25 m good first guess for robots with ~1m
          reach.
        :param offset: Offset of the grid in the world frame, i.e., its center.
        :param goal_orientation_sampler: Sampler for goal orientations.
        :param id_prefix: Prefix for task IDs.
        :param rng: Random number generator.
        :param base_tasks: List of base tasks to mutate; all of these are archtypes of obstacles and eef poses defined
          above this class.
        :param diff_level: Difficulty level of the task; either "hard" or "medium".
        """
        super().__init__(id_prefix=id_prefix, rng=rng)
        self.base_tasks = base_tasks
        self._initial_state = self.rng.bit_generator.state
        if isinstance(grid_size, (int, np.integer)):
            grid_size = (grid_size, grid_size, grid_size)
        if isinstance(spacing, float):
            spacing = (spacing, spacing, spacing)
        self._grid_sampler = CartesianGridSampler(grid_size=grid_size, spacing=spacing, offset=offset,
                                                  rotation_sampler=goal_orientation_sampler, rng=self.rng)
        self._index_sampler = GridIndexSampler(grid_size=grid_size, rng=self.rng)
        self._center_goals: bool = True
        self.diff_level = diff_level

    def _rotate_90_degrees_x_axis(self, objects):
        """Helper to rotate objects, s.a. goals and obstacles, 90 degrees around the x-axis"""
        order = ["x", "y", "z"]
        rotated_objects = []
        for obj in objects:
            rotated_obj = tuple(
                obj[order.index(dim)] for dim in ["x", "y", "z"]
            )
            rotated_objects.append(rotated_obj)
        return rotated_objects

    def _mutate_task(self, task: Task):
        """
        Add shift and/or rotation to all goals and obstacles in the task; diff_level determines range of shifts and

        rotations.
        """
        # (x - red, y - green, z - blue)
        R = task[2]
        if self.diff_level == "hard":
            dx = self.rng.choice([9, 8, 1, 0, -1])
            dy = self.rng.integers(-2, 3)
            dz = self.rng.integers(-2, 4)
            dx_g = self.rng.integers(-1, 1)
            dy_g = self.rng.integers(-2, 2)
            dz_g = self.rng.integers(-1, 2)
        else:
            dx = self.rng.choice([9, 8, 7, 1, 0])
            dy = self.rng.integers(-3, 1)
            dz = self.rng.integers(-2, 3)
            dx_g = self.rng.integers(-1, 1)
            dy_g = self.rng.integers(0, 2)
            dz_g = self.rng.integers(-1, 2)

        goals = [(x + dx, y + dy, z + dz) if (x, y, z) == task[0][-1]
                 else (x + dx + dx_g, y + dy + dy_g, z + dz + dz_g)
                 for x, y, z in task[0]]
        goals = tuple(goals)

        obstacles = [(x + dx, y + dy, z + dz) for x, y, z in task[1]]
        obstacles = tuple(obstacles)

        if self.rng.random() < 0.5:
            obstacles = self._rotate_90_degrees_x_axis(obstacles)
            goals = self._rotate_90_degrees_x_axis(goals)
            z_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            R = np.dot(task[2], z_rot)

        return goals, obstacles, R

    def __next__(self) -> Task:
        """Samples a new task with non-overlapping goals and obstacles"""
        task_name = self.rng.choice(self.base_tasks)
        goals_idx, obstacles_idx, rot_mat = self._mutate_task(HARD_TASKS[task_name])
        goals = list()
        for i, idx in enumerate(goals_idx):
            p = self._grid_sampler.grid_idx2grid_value(idx)
            if not self._center_goals:
                p += self._grid_sampler.spacing * (1 - 2 * self.rng.random(3))
            # rot_mat = self._grid_sampler.sample_orientation()
            T = self._grid_sampler.sample_2_transformation(orientation=rot_mat, translation=p)
            desired = ToleratedPose(T)
            goals.append(Reach(f'Goal {i}', desired))
        obstacles = list()
        for i, idx in enumerate(obstacles_idx):
            p = self._grid_sampler.grid_idx2grid_value(idx)
            T = self._grid_sampler.sample_2_transformation(orientation=np.eye(3), translation=p)
            geometry = self._make_box(T)
            obstacles.append(Obstacle(f'Obstacle {i}', geometry))

        task = Task(self.get_task_header(), obstacles=obstacles, goals=goals)
        task.constraints = list(c for c in task.constraints
                                if type(c) not in {SelfCollisionFree, BasePlacement})
        task.constraints.append(BasePlacement(ToleratedPose(
            Transformation.neutral(),
            CartesianXYZ((-1, 1), (-1, 1), (-1, 1)))))

        return task

    def _make_box(self, placement: Transformation) -> Box:
        """Creates a box that fills one grid element"""
        spacing = self._grid_sampler._grid_spacing
        return Box({'x': spacing[0], 'y': spacing[1], 'z': spacing[2]}, pose=placement)


if __name__ == '__main__':
    print("Creating edge-case dataset for base-pose optimization")
    for obstacle_type, diff_level in itertools.product(HARD_TASKS.keys(), ["hard", "medium"]):
        print(f"Creating tasks for {obstacle_type} -- {diff_level}")
        task_gen = HardTaskGenerator(grid_size=5, spacing=.25, base_tasks=[obstacle_type, ],
                                     diff_level=diff_level, rng=np.random.default_rng(42))
        for i, task in enumerate(itertools.islice(task_gen, 10)):
            task.header = TaskHeader(
                f"base_opt/edge_case/{obstacle_type}_{diff_level}_{i}",
                date=datetime(2024, 7, 8, 12, 0, 0),
                tags=["BPO24", "base pose optimization 2024", "Any base rotation", "edge_case",
                      obstacle_type, f"edge_case_{diff_level}", "3 goals"],
                author=["Friedrich Dang", "Matthias Mayer"],
                email=["friedrich.dang@tum.de", "matthias.mayer@tum.de"],
                affiliation=2 * ["Technical University of Munich"],
                version="2022"
            )
            task.to_json_file(DATA.joinpath("tasks"))
