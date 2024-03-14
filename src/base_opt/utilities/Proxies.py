from typing import Any, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from mcs.environments.Proxies import FixedPenaltyTaskSolverAssemblyEvaluationEnvironment
from timor import ModuleAssembly, Transformation
from timor.task import Constraints
from timor.task.CostFunctions import CostFunctionBase
from timor.task.Goals import At
from timor.task.Task import Task
from timor.task.Tolerance import CartesianXYZ, Composed
from timor.utilities import logging
from timor.utilities.transformation import Projection


def base_constraint2cartesian_xyz(base_constraint: Constraints.BasePlacement) -> CartesianXYZ:
    """Extract CartesianXYZ from base constraint. Raises ValueError if not possible."""
    tolerance = base_constraint.base_pose.tolerance
    if not isinstance(tolerance, CartesianXYZ):
        if isinstance(tolerance, Composed):
            cart_xyz = \
                [t for t in tolerance.tolerances if isinstance(t, CartesianXYZ)]
            if len(cart_xyz) > 1:
                raise ValueError("BaseChangeEnvironment only supports a single CartesianXYZ tolerance within a "
                                 "composed tolerance")
            elif len(tolerance.tolerances) == 0:
                logging.info("Tolerance seem empty; default to +/- 1m in each translation direction.")
                cart_xyz = [CartesianXYZ((-1., 1.), (-1., 1.), (-1., 1.)), ]
            else:
                logging.info("Only considering first CartesianXYZ tolerance for base constraint.")
            tolerance = cart_xyz[0]
        else:
            raise ValueError(f"base_constraint2cartesian_xyz does not support tolerance of type:"
                             f" {type(tolerance)}")
    return tolerance


def xyz_to_transformation(action: np.ndarray, base_constraint: Constraints.BasePlacement) -> Transformation:
    """
    Map an action in [-1, 1]^3 to a base pose within the base constraint.

    The action is x, y, z, where -1 = left boundary of base constraint, +1 = right boundary of base constraint.
    """
    action = (action + 1) / 2  # Scale to [0, 1]
    bounds = base_constraint2cartesian_xyz(base_constraint)
    bounds = bounds.stacked
    desired_offset = bounds[:, 0] + action * (bounds[:, 1] - bounds[:, 0])  # Lin. interpolation
    return (base_constraint.base_pose.nominal.in_world_coordinates() @
            Transformation.from_translation(desired_offset))


def xyz_rotvec_to_transformation(action: np.ndarray, base_constraint: Constraints.BasePlacement) -> Transformation:
    """
    Map an action in [-1, 1]^6 to a base pose within the base constraint.

    The action is x, y, z, theta * (n_x, n_y, n_z); x, y, z are handled as in xyz_to_transformation.
    The rotation is _not_ limited to the base constraint, but -1 = rotation by -pi in this axis, +1 = rotation by pi
    """
    translation = xyz_to_transformation(action[:3], base_constraint)
    rotation = Rotation.from_rotvec(action[3:] * np.pi)  # Any rotation vector up to pi about arbitrary axis
    return translation @ Transformation.from_rotation(rotation.as_matrix())


class BaseChangeEnvironment(FixedPenaltyTaskSolverAssemblyEvaluationEnvironment):
    """
    An environment that takes pre-built assemblies and allows to change the base pose via actions.

    Default:
      * Actions (so far) only vary the base position within the task's base constraint range (-1 = lower bound in each
        projection, +1 = upper bound in each projection).
      * Observations include the roto_translation_vector relative to the nominal base pose. Distances here are _not_
        re-scaled to [-1, 1] but are in meters, expecting that this is also the order of magnitude of the assembly and
        overall task.

    Other observations/actions are configurable via pose_projection and action2base_pose.
    """

    # 3D translation in between -1 (lower limit of base constraint) and 1 (upper limit of base constraint)
    action_space = gym.spaces.box.Box(low=-1., high=1., shape=(3,))
    # Observation is stacked goal poses
    observation_space = gym.spaces.dict.Dict({})

    available_actions = {
        'xyz': (
            xyz_to_transformation,
            gym.spaces.box.Box(low=-1., high=1., shape=(3,))
        ),
        'xyz_rotvec': (
            xyz_rotvec_to_transformation,
            gym.spaces.box.Box(low=-1., high=1., shape=(6,))
        )
    }

    def __init__(self,
                 assembly: ModuleAssembly,
                 task_solver: 'TaskSolverBase',
                 cost_function: CostFunctionBase,
                 task: Optional[Task] = None,
                 render_mode: str = 'human',
                 reward_fail: float = -1000.0,
                 pose_projection: str = 'roto_translation_vector',
                 action2base_pose: str = 'xyz',
                 observations: Iterable[str] = ('goal_poses',),):
        """
        Init. esp. make assembly static

        :param assembly: The robot assembly to use for evaluation.
        :param task_solver: The task solver to use for evaluation.
        :param cost_function: The cost function to use for evaluation of reward (negative cost).
        :param task: The task to use for evaluation. If None, the task of the task solver will be used.
          Note that the task solver's task will be modified to have a wider collision margin.
        :param render_mode: The render mode to use for visualization.
        :param reward_fail: The reward to return if the task solver fails.
        :param pose_projection: The projection to use for the observation of the goal pose.
          Defaults to roto_translation_vector and can use any :class:`timor.utilities.Transformation.Projection`.
        :param action2base_pose: A str to describe the mapping from an action to a base pose within the base constraint,
          see available_actions for options.
        """
        super().__init__(reward_fail=reward_fail, task_solver=task_solver, cost_function=cost_function, task=task,
                         render_mode=render_mode)
        if any(not isinstance(g, At) for g in self.task.goals):
            raise ValueError("BaseChangeEnvironment only supports At goals")
        if not hasattr(Projection, pose_projection):
            raise ValueError(f"Projection {pose_projection} not supported by Projection class.")

        self.assembly = assembly

        if action2base_pose not in self.available_actions:
            raise ValueError(f"Unknown action2base_pose {action2base_pose}. "
                             f"Choose from {list(self.available_actions.keys())}")
        self._action2base_pose, self.action_space = self.available_actions[action2base_pose]
        self.selected_action2base_pose = action2base_pose
        logging.info(f"Set action space to {self.action_space} due to action2base_pose {self.action2base_pose}")
        self.reset()

    def _calc_obs(self) -> Dict[str, np.ndarray]:
        """Calculate the observation dict."""
        return {}

    def action2base_pose(self, action: np.ndarray):
        """Map an action to a base pose within the base constraint."""
        try:
            base_constraint = self.task.base_constraint
        except AttributeError:
            raise ValueError("Base constraint must be defined for BaseChangeEnvironment")
        return self._action2base_pose(action, base_constraint)

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[dict[str, np.ndarray], dict]:
        """Reset the environment and get first observation (stacked goal poses as roto translation vector)."""
        assembly_tmp = self.assembly
        super().reset(seed=seed, options=options)
        self.assembly = assembly_tmp
        if self.assembly is not None:
            self.assembly.robot.update_configuration(np.zeros_like(self.assembly.robot.configuration))

        return self._calc_obs(), {}

    def step(self, action: np.ndarray,
             detailed_info: bool = False) -> Tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """
        Adapt to AssemblyEvaluationEnvironment by just changing the base pose of the robot and using desired goal poses

        as observation.

        :param action: CartesianXYZ action in [-1, 1] to be mapped to base pose within base constraint.
        :param detailed_info: If true, additional information such as the solution object is returned in the info dict
          (!takes considerable effort to (de)serialize with gym).
        """
        desired_pose = self.action2base_pose(action)
        self.assembly.robot.set_base_placement(desired_pose)
        if __debug__ and self.assembly.robot.placement != self.action2base_pose(action):
            logging.warning("Robot placement was not set correctly")
        sol, reward, end, truncated, info = super().step(self.assembly)
        if __debug__ and self.assembly.robot.placement != self.action2base_pose(action):
            logging.warning(f"Inner task solve changed base pose. "
                            f"Desired: {self.action2base_pose(action)}, actual: {self.assembly.robot.placement}")
        if __debug__ and sol is not None and sol.robot.placement != self.action2base_pose(action):
            logging.warning(f"Solution base pose != desired chosen"
                            f"Desired: {self.action2base_pose(action)}, actual: {sol.robot.placement}")
        return self._calc_obs(), reward, end, truncated, info | ({'solution': sol} if detailed_info else {})
