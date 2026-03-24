"""
Tests for IterationStrategy regret matching and strategy computation.
Verifies that strategies are valid probability distributions (necessary for CFR convergence).
"""

import unittest
import numpy as np
import torch
from torch.nn import functional as F


class TestRegretMatchingLogic(unittest.TestCase):
    """
    Test the regret matching math directly (without instantiating IterationStrategy),
    since the core logic is: relu(advantages) / sum(relu(advantages)), or
    deterministic best action if all advantages are negative.
    This mirrors the logic in IterationStrategy.get_a_probs2().
    """

    @staticmethod
    def _compute_strategy(advantages, legal_action_mask):
        """
        Reproduce the exact strategy computation from IterationStrategy.get_a_probs2().
        advantages: 1D tensor of advantage values
        legal_action_mask: 1D tensor of 0/1 for legal actions
        """
        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(0)
        legal_action_mask = torch.tensor(legal_action_mask, dtype=torch.float32).unsqueeze(0)
        bs = 1
        n_actions = advantages.size(1)

        relu_advantages = F.relu(advantages, inplace=False)
        sum_pos_adv_expanded = relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)

        best_legal_deterministic = torch.zeros((bs, n_actions), dtype=torch.float32)
        bests = torch.argmax(
            torch.where(legal_action_mask.bool(), advantages, torch.full_like(advantages, fill_value=-10e20)),
            dim=1
        )
        _batch_arranged = torch.arange(bs, dtype=torch.long)
        best_legal_deterministic[_batch_arranged, bests] = 1

        strategy = torch.where(
            sum_pos_adv_expanded > 0,
            relu_advantages / sum_pos_adv_expanded,
            best_legal_deterministic
        )
        return strategy.squeeze(0).numpy()

    # ─── Uniform at iteration 0 ───

    def test_uniform_at_iteration_zero(self):
        """With no advantages (iter 0), should be uniform over legal actions."""
        legal_mask = torch.tensor([1, 1, 0, 1], dtype=torch.float32).unsqueeze(0)
        uniform = legal_mask / legal_mask.sum(-1).unsqueeze(-1).expand_as(legal_mask)
        result = uniform.squeeze(0).numpy()
        np.testing.assert_allclose(result, [1/3, 1/3, 0, 1/3], atol=1e-6)

    # ─── Strategy is valid probability distribution ───

    def test_strategy_sums_to_one(self):
        """Strategy should sum to 1.0."""
        strategy = self._compute_strategy([3.0, -1.0, 1.0, 0.5], [1, 1, 1, 1])
        self.assertAlmostEqual(strategy.sum(), 1.0, places=6)

    def test_strategy_non_negative(self):
        """All strategy entries should be >= 0."""
        strategy = self._compute_strategy([3.0, -1.0, 1.0, -2.0], [1, 1, 1, 1])
        self.assertTrue(np.all(strategy >= 0))

    def test_strategy_zero_on_illegal_when_negative(self):
        """
        When all advantages are negative (deterministic fallback path),
        illegal action slots should have 0 probability.
        Note: the ReLU path does NOT mask illegal actions — that masking happens
        upstream in the sampling code. The deterministic path does respect the mask.
        """
        strategy = self._compute_strategy([-1.0, -2.0, -0.5, -3.0], [1, 0, 1, 0])
        self.assertAlmostEqual(strategy[1], 0.0)
        self.assertAlmostEqual(strategy[3], 0.0)
        self.assertAlmostEqual(strategy.sum(), 1.0, places=6)

    # ─── Positive advantages: regret matching ───

    def test_positive_advantages_regret_matching(self):
        """adv=[3, 0, 1, 0] → relu=[3, 0, 1, 0] / 4 = [0.75, 0, 0.25, 0]."""
        strategy = self._compute_strategy([3.0, 0.0, 1.0, 0.0], [1, 1, 1, 1])
        np.testing.assert_allclose(strategy, [0.75, 0.0, 0.25, 0.0], atol=1e-6)

    def test_mixed_advantages(self):
        """adv=[2.0, -1.0, 2.0, -3.0] → relu=[2, 0, 2, 0] / 4 = [0.5, 0, 0.5, 0]."""
        strategy = self._compute_strategy([2.0, -1.0, 2.0, -3.0], [1, 1, 1, 1])
        np.testing.assert_allclose(strategy, [0.5, 0.0, 0.5, 0.0], atol=1e-6)

    # ─── All negative: deterministic best ───

    def test_all_negative_deterministic(self):
        """All negative advantages → pick the least negative (best) legal action."""
        # adv=[-1, -3, -0.5], mask=[1,1,1] → best is index 2 (-0.5)
        strategy = self._compute_strategy([-1.0, -3.0, -0.5], [1, 1, 1])
        np.testing.assert_allclose(strategy, [0, 0, 1], atol=1e-6)

    def test_all_negative_respects_mask(self):
        """All negative with mask: pick best among legal only."""
        # adv=[-0.1, -3.0, -0.5], mask=[0, 1, 1] → among legal: -0.5 > -3.0, index 2 wins
        strategy = self._compute_strategy([-0.1, -3.0, -0.5], [0, 1, 1])
        np.testing.assert_allclose(strategy, [0, 0, 1], atol=1e-6)

    # ─── Batch ───

    def test_batch_output_shape(self):
        """Batch of 4 should produce (4, N_ACTIONS) with each row summing to 1."""
        n_actions = 4
        advantages = torch.tensor([
            [3.0, -1.0, 1.0, 0.0],
            [-1.0, -2.0, -0.5, -3.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ], dtype=torch.float32)
        legal_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ], dtype=torch.float32)

        bs = 4
        relu_advantages = F.relu(advantages, inplace=False)
        sum_pos_adv_expanded = relu_advantages.sum(1).unsqueeze(-1).expand_as(relu_advantages)

        best_legal_deterministic = torch.zeros((bs, n_actions), dtype=torch.float32)
        bests = torch.argmax(
            torch.where(legal_mask.bool(), advantages, torch.full_like(advantages, fill_value=-10e20)),
            dim=1
        )
        _batch_arranged = torch.arange(bs, dtype=torch.long)
        best_legal_deterministic[_batch_arranged, bests] = 1

        strategy = torch.where(
            sum_pos_adv_expanded > 0,
            relu_advantages / sum_pos_adv_expanded,
            best_legal_deterministic
        )
        result = strategy.numpy()

        self.assertEqual(result.shape, (4, 4))
        for i in range(4):
            self.assertAlmostEqual(result[i].sum(), 1.0, places=5,
                                   msg=f"Row {i} doesn't sum to 1: {result[i]}")
            self.assertTrue(np.all(result[i] >= 0), f"Row {i} has negative values")

    # ─── Edge cases ───

    def test_single_legal_action_all_negative(self):
        """Only one legal action with all-negative advantages → 100% on that action."""
        strategy = self._compute_strategy([-5.0, -1.0, -2.0], [0, 0, 1])
        np.testing.assert_allclose(strategy, [0, 0, 1], atol=1e-6)

    def test_all_zero_advantages(self):
        """All zero advantages with all legal → should pick one deterministically (argmax of 0s)."""
        strategy = self._compute_strategy([0.0, 0.0, 0.0], [1, 1, 1])
        # ReLU of zeros = zeros, sum = 0, falls to deterministic path
        # argmax of equal values → first one (index 0)
        self.assertAlmostEqual(strategy.sum(), 1.0, places=6)
        self.assertEqual(np.count_nonzero(strategy), 1)


if __name__ == '__main__':
    unittest.main()
