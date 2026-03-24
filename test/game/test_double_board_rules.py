"""
Tests for Double Board Bomb Pot PLO rules constants and game class configuration.
"""

import unittest

from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules_double_board_bomb_pot import DoubleBoardBombPotPLORules
from PokerRL.game.games import DoubleBoardBombPotPLO


class TestDoubleBoardBombPotRules(unittest.TestCase):
    """Verify the rules class constants are correct and consistent."""

    def test_no_preflop_round(self):
        """Bomb pots skip preflop — only FLOP, TURN, RIVER."""
        self.assertNotIn(Poker.PREFLOP, DoubleBoardBombPotPLORules.ALL_ROUNDS_LIST)
        self.assertEqual(DoubleBoardBombPotPLORules.ALL_ROUNDS_LIST,
                         [Poker.FLOP, Poker.TURN, Poker.RIVER])

    def test_two_boards_ten_cards(self):
        """Two independent boards with 5 cards each = 10 total."""
        self.assertEqual(DoubleBoardBombPotPLORules.N_BOARDS, 2)
        self.assertEqual(DoubleBoardBombPotPLORules.N_TOTAL_BOARD_CARDS, 10)

    def test_round_before_mapping(self):
        """FLOP maps to itself (no preflop before it), TURN→FLOP, RIVER→TURN."""
        rb = DoubleBoardBombPotPLORules.ROUND_BEFORE
        self.assertEqual(rb[Poker.FLOP], Poker.FLOP)
        self.assertEqual(rb[Poker.TURN], Poker.FLOP)
        self.assertEqual(rb[Poker.RIVER], Poker.TURN)

    def test_round_after_mapping(self):
        """FLOP→TURN, TURN→RIVER, RIVER→None (game ends)."""
        ra = DoubleBoardBombPotPLORules.ROUND_AFTER
        self.assertEqual(ra[Poker.FLOP], Poker.TURN)
        self.assertEqual(ra[Poker.TURN], Poker.RIVER)
        self.assertIsNone(ra[Poker.RIVER])

    def test_inherits_plo_properties(self):
        """Bomb pot rules inherit PLO fundamentals: 4 hole cards, suits matter."""
        self.assertEqual(DoubleBoardBombPotPLORules.N_HOLE_CARDS, 4)
        self.assertTrue(DoubleBoardBombPotPLORules.SUITS_MATTER)

    def test_preflop_not_in_round_mappings(self):
        """PREFLOP should not appear as a key in ROUND_BEFORE or ROUND_AFTER."""
        self.assertNotIn(Poker.PREFLOP, DoubleBoardBombPotPLORules.ROUND_BEFORE)
        self.assertNotIn(Poker.PREFLOP, DoubleBoardBombPotPLORules.ROUND_AFTER)


class TestDoubleBoardBombPotPLOGameClass(unittest.TestCase):
    """Verify the concrete game class configuration."""

    def test_game_class_blinds(self):
        """Bomb pots have no blinds — only antes."""
        self.assertEqual(DoubleBoardBombPotPLO.SMALL_BLIND, 0)
        self.assertEqual(DoubleBoardBombPotPLO.BIG_BLIND, 0)

    def test_game_class_ante(self):
        """ANTE should be 300 (3bb equivalent)."""
        self.assertEqual(DoubleBoardBombPotPLO.ANTE, 300)

    def test_game_class_pot_limit(self):
        """Double board bomb pot is a pot-limit game."""
        self.assertTrue(DoubleBoardBombPotPLO.IS_POT_LIMIT_GAME)
        self.assertFalse(DoubleBoardBombPotPLO.IS_FIXED_LIMIT_GAME)

    def test_game_class_fixed_flops(self):
        """Fixed flop configuration should match expected cards."""
        # Board 1: Ks 7h 2d = [11,2], [5,0], [0,1]
        self.assertEqual(DoubleBoardBombPotPLO.FIXED_FLOP_BOARD1,
                         [[11, 2], [5, 0], [0, 1]])
        # Board 2: Qh Js 9c = [10,0], [9,2], [7,3]
        self.assertEqual(DoubleBoardBombPotPLO.FIXED_FLOP_BOARD2,
                         [[10, 0], [9, 2], [7, 3]])

    def test_default_stack_size(self):
        self.assertEqual(DoubleBoardBombPotPLO.DEFAULT_STACK_SIZE, 10000)

    def test_ev_normalizer(self):
        """EV normalizer = 1000 / ANTE (milli-antes)."""
        expected = 1000.0 / 300
        self.assertAlmostEqual(DoubleBoardBombPotPLO.EV_NORMALIZER, expected)

    def test_win_metric(self):
        self.assertEqual(DoubleBoardBombPotPLO.WIN_METRIC, Poker.MeasureAnte)


if __name__ == '__main__':
    unittest.main()
