"""
Tests for AbstractedCFR on Double Board Bomb Pot PLO.
Uses random bucket assignments to skip expensive equity computation.
"""

import unittest
import tempfile
import shutil
import numpy as np

from PokerRL.game.games import DoubleBoardBombPotPLO
from PokerRL.game import bet_sets
from PokerRL.game.card_abstraction import CardAbstraction
from PokerRL.cfr.AbstractedCFR import AbstractedCFR


class FastRandomAbstraction(CardAbstraction):
    """
    Test-only abstraction that assigns random buckets instantly.
    Skips the expensive Monte Carlo equity computation.
    """

    def _compute_preflop_buckets(self):
        return np.random.randint(0, self._n_buckets, size=self._rules.RANGE_SIZE, dtype=np.int16)

    def _compute_postflop_buckets(self, board_2d):
        valid_mask = self.get_blocked_mask(board_2d)
        buckets = np.full(self._rules.RANGE_SIZE, -1, dtype=np.int16)
        valid_indices = np.where(valid_mask)[0]
        buckets[valid_indices] = np.random.randint(0, self._n_buckets, size=len(valid_indices), dtype=np.int16)
        return buckets


class TestAbstractedCFR_DBBP(unittest.TestCase):
    """Smoke tests for AbstractedCFR on Double Board Bomb Pot PLO."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _make_cfr(self, n_buckets=10, variant='plus'):
        """Build CFR with fast random abstraction for testing."""
        cfr = AbstractedCFR(
            name='test_dbbp',
            game_cls=DoubleBoardBombPotPLO,
            agent_bet_set=bet_sets.POT_ONLY,
            n_buckets=n_buckets,
            variant=variant,
            n_rollouts=10,
            cache_dir=self.tmpdir,
        )
        # Swap in fast random abstraction
        cfr._card_abs = FastRandomAbstraction(
            rules=cfr._env_bldr.rules,
            lut_holder=cfr._env_bldr.lut_holder,
            n_buckets=n_buckets,
            n_rollouts=10,
            cache_dir=self.tmpdir,
        )
        cfr._tree._card_abs = cfr._card_abs
        cfr._tree._strategy_filler._card_abs = cfr._card_abs
        cfr._tree._value_filler._card_abs = cfr._card_abs
        return cfr

    def test_reset_builds_tree(self):
        """reset() should build the game tree without errors."""
        cfr = self._make_cfr()
        cfr.reset()
        self.assertGreater(cfr._tree.n_nodes, 0)
        print("Tree has {} nodes".format(cfr._tree.n_nodes))

    def test_iteration_runs(self):
        """A single CFR iteration should complete without errors."""
        cfr = self._make_cfr()
        cfr.reset()
        cfr.iteration()

    def test_exploitability_finite(self):
        """Exploitability should be a finite number after iterations."""
        cfr = self._make_cfr()
        cfr.reset()
        for _ in range(3):
            cfr.iteration()
        expl = cfr.get_exploitability()
        self.assertNotEqual(expl, float('inf'))
        import math
        self.assertFalse(math.isnan(expl))

    def test_strategies_valid(self):
        """Strategies should be valid probability distributions."""
        cfr = self._make_cfr()
        cfr.reset()
        for _ in range(3):
            cfr.iteration()

        def _check(node):
            if not node.is_terminal and node.p_id_acting_next != cfr._tree.CHANCE_ID:
                row_sums = node.strategy.sum(axis=1)
                for b in range(cfr._n_buckets):
                    self.assertAlmostEqual(row_sums[b], 1.0, places=3)
                self.assertTrue((node.strategy >= -1e-6).all())
            for c in node.children:
                _check(c)

        _check(cfr._tree.root)

    def test_save_strategy(self):
        """Strategy should save to disk without error."""
        import os
        cfr = self._make_cfr()
        cfr.reset()
        cfr.iteration()
        path = os.path.join(self.tmpdir, 'test_strat.npz')
        cfr.save_strategy(path)
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
