"""
Tests of the various optimizers implemented in base_opt.

:author: Matthias Mayer
:date: 15.12.23
"""
from collections import namedtuple
import unittest

import numpy as np
import numpy.testing as np_test

from base_opt.base_opt.BaseOptimizer import BaseOptimizationHistory


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
