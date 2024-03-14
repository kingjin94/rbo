from typing import Iterable, Optional

import numpy as np
from timor.task import Constraints
from timor.task.Task import Task

from mcs.optimize.Sampler import FiniteSetSampler
from mcs.optimize.TaskGenerator import TaskGenerator


class FixedSetTaskGenerator(TaskGenerator):
    """A task generator returning tasks from a fixed set of tasks."""

    def __init__(self,
                 rng: Optional[np.random.Generator] = None,
                 tasks: Iterable[Task] = (),
                 constraints: Optional[Iterable[Constraints.ConstraintBase]] = None):
        """
        A finite set task generator yields a fixed set of tasks.

        :param rng: A random number generator
        :param tasks: The tasks to be yielded
        :param constraints: Constraints that are applied to every task; if None is given, the constraints of the tasks
          are kept as is.
        """
        super().__init__(rng=rng)
        self._constraints = constraints
        self._sampler = FiniteSetSampler(tasks, rng=rng)

    def __next__(self):
        """Returns the next task from the set"""
        task = self._sampler.__next__()
        if self._constraints is not None:
            task.constraints = self._constraints
        return task

    def __len__(self):
        """Returns the number of tasks in the set"""
        return len(self._sampler)

    def as_finite_iterable(self) -> Iterable[Task]:
        """Returns an iterator over all tasks"""
        return self._sampler.as_finite_iterable()
