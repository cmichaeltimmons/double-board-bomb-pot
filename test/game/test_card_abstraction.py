"""
Tests for card abstraction module.
"""

import unittest
import numpy as np
import tempfile
import shutil

from PokerRL.game._.rl_env.game_rules_plo import PLORules
from PokerRL.game._.look_up_table import LutHolderPLO
from PokerRL.game.card_abstraction import CardAbstraction


class TestCardAbstraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rules = PLORules
        cls.lut = LutHolderPLO(PLORules)
        cls.tmpdir = tempfile.mkdtemp()
        # Use small rollouts for fast tests
        cls.abs = CardAbstraction(
            rules=cls.rules, lut_holder=cls.lut,
            n_buckets=10, n_rollouts=50, cache_dir=cls.tmpdir
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_blocked_mask_no_board(self):
        """No board cards => all hands valid."""
        board = np.array([], dtype=np.int8).reshape(0, 2)
        mask = self.abs.get_blocked_mask(board)
        self.assertEqual(mask.shape[0], self.rules.RANGE_SIZE)
        self.assertTrue(np.all(mask))

    def test_blocked_mask_with_board(self):
        """Board cards should block hands containing those cards."""
        # Board: Ah, Kd, Qs (cards 48, 47, 42 in some mapping)
        board = np.array([[12, 0], [11, 1], [10, 2]], dtype=np.int8)
        mask = self.abs.get_blocked_mask(board)
        self.assertEqual(mask.shape[0], self.rules.RANGE_SIZE)

        # Some hands should be blocked
        n_valid = np.sum(mask)
        self.assertGreater(n_valid, 0)
        self.assertLess(n_valid, self.rules.RANGE_SIZE)

        # Verify: any hand with Ah (rank=12, suit=0) should be blocked
        ah_1d = int(self.lut.LUT_2DCARD_2_1DCARD[12, 0])
        hole_cards = self.lut.LUT_IDX_2_HOLE_CARDS
        for ridx in range(min(1000, self.rules.RANGE_SIZE)):
            if ah_1d in hole_cards[ridx]:
                self.assertFalse(mask[ridx],
                                 "Hand {} contains Ah but is not blocked".format(ridx))

    def test_preflop_buckets_shape(self):
        """Preflop buckets should have correct shape and range."""
        buckets = self.abs.get_preflop_buckets()
        self.assertEqual(buckets.shape, (self.rules.RANGE_SIZE,))
        # All hands should have a valid bucket (preflop, nothing blocked)
        self.assertTrue(np.all(buckets >= 0))
        self.assertTrue(np.all(buckets < self.abs.n_buckets))

    def test_preflop_buckets_deterministic(self):
        """Same abstraction should produce same buckets."""
        b1 = self.abs.get_preflop_buckets()
        b2 = self.abs.get_preflop_buckets()
        np.testing.assert_array_equal(b1, b2)

    def test_preflop_all_buckets_used(self):
        """All bucket IDs should be assigned to at least one hand."""
        buckets = self.abs.get_preflop_buckets()
        unique = np.unique(buckets[buckets >= 0])
        self.assertEqual(len(unique), self.abs.n_buckets)

    def test_preflop_isomorphic_hands_same_bucket(self):
        """Hands with same rank pattern but different suits should get same bucket."""
        buckets = self.abs.get_preflop_buckets()
        hole_cards = self.lut.LUT_IDX_2_HOLE_CARDS
        card_2d = self.lut.LUT_1DCARD_2_2DCARD

        # Find two hands with same rank pattern
        # Hand 0: first combination. Find another with same ranks, different suits.
        h0_cards = hole_cards[0]
        h0_ranks = tuple(sorted(int(card_2d[c][0]) for c in h0_cards))

        found_match = False
        for ridx in range(1, self.rules.RANGE_SIZE):
            h_cards = hole_cards[ridx]
            h_ranks = tuple(sorted(int(card_2d[c][0]) for c in h_cards))
            if h_ranks == h0_ranks:
                # Same rank pattern
                self.assertEqual(buckets[0], buckets[ridx],
                                 "Hands with same rank pattern have different buckets")
                found_match = True
                break

        self.assertTrue(found_match, "Could not find hand with same rank pattern")

    def test_postflop_buckets_blocked_hands(self):
        """Blocked hands should have bucket -1 in postflop."""
        board = np.array([[12, 0], [11, 1], [10, 2]], dtype=np.int8)
        buckets = self.abs.get_postflop_buckets(board)
        mask = self.abs.get_blocked_mask(board)

        # Blocked hands should have bucket -1
        self.assertTrue(np.all(buckets[~mask] == -1))
        # Valid hands should have valid buckets
        self.assertTrue(np.all(buckets[mask] >= 0))
        self.assertTrue(np.all(buckets[mask] < self.abs.n_buckets))

    def test_postflop_cache_roundtrip(self):
        """Cached and fresh computations should match."""
        board = np.array([[5, 0], [8, 1], [3, 2]], dtype=np.int8)
        b1 = self.abs.get_postflop_buckets(board)
        # Clear in-memory cache
        key = self.abs._board_hash(board)
        del self.abs._postflop_cache[key]
        # Should load from disk cache
        b2 = self.abs.get_postflop_buckets(board)
        np.testing.assert_array_equal(b1, b2)

    def test_percentile_bucket_uniform(self):
        """Percentile bucketing with uniform data should spread evenly."""
        equity = np.linspace(0, 1, 1000).astype(np.float32)
        valid = np.ones(1000, dtype=bool)
        abs_small = CardAbstraction(
            rules=type('R', (), {'RANGE_SIZE': 1000, 'N_CARDS_IN_DECK': 52,
                                  'N_HOLE_CARDS': 4})(),
            lut_holder=self.lut,
            n_buckets=10, n_rollouts=10, cache_dir=self.tmpdir
        )
        buckets = abs_small._percentile_bucket(equity, valid)
        # Each bucket should have roughly 100 hands
        for b in range(10):
            count = np.sum(buckets == b)
            self.assertGreater(count, 50)
            self.assertLess(count, 150)


if __name__ == '__main__':
    unittest.main()
