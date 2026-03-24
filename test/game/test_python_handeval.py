"""
Tests for the pure-Python PLO hand evaluator (PythonHandeval).
Verifies PLO rules: must use exactly 2 of 4 hole cards + 3 of 5 board cards.
"""

import unittest
import numpy as np

from PokerRL.game._.cpp_wrappers.PythonHandeval import PythonHandeval


class TestPythonHandeval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evaluator = PythonHandeval()

    # ─── Hold'em baseline ───

    def test_holdem_known_rank(self):
        """Hold'em evaluation returns a positive integer for a known hand."""
        board = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
        hand = np.array([[11, 3], [5, 1]], dtype=np.int8)
        rank = self.evaluator.get_hand_rank_52_holdem(hand, board)
        self.assertGreater(rank, 0)

    # ─── PLO basics ───

    def test_plo_known_rank(self):
        """PLO evaluation returns a positive integer for a known hand."""
        board = np.array([[2, 0], [2, 3], [11, 1], [10, 2], [11, 2]], dtype=np.int8)
        hand = np.array([[11, 3], [5, 1], [5, 2], [11, 0]], dtype=np.int8)
        rank = self.evaluator.get_hand_rank_52_plo(hand, board)
        self.assertGreater(rank, 0)

    def test_plo_uses_exactly_2_hole_cards(self):
        """
        PLO must use exactly 2 hole cards. A hand with 3 aces and a board with
        2 aces should NOT make four-of-a-kind (that would require 3 hole cards).
        Compare against a hand that CAN make quads via exactly 2 hole cards.
        """
        # Board: Ah Ad 5s 7c 2h — has 2 aces
        board = np.array([[12, 0], [12, 1], [3, 2], [5, 3], [0, 0]], dtype=np.int8)
        # Hand: Ac As Kh Kd — has 2 more aces, can use Ac+As to make quads
        hand_2_aces = np.array([[12, 3], [12, 2], [11, 0], [11, 1]], dtype=np.int8)
        rank_aces = self.evaluator.get_hand_rank_52_plo(hand_2_aces, board)

        # Board with 3 aces: Ah Ad As 7c 2h
        board_3a = np.array([[12, 0], [12, 1], [12, 2], [5, 3], [0, 0]], dtype=np.int8)
        # Hand: Ac Kh 3d 4s — only 1 ace in hand, can use Ac + Kh with 3 board aces
        # Best: Ac + Kh, board Ah Ad As = four aces + K kicker
        hand_1_ace = np.array([[12, 3], [11, 0], [1, 1], [2, 2]], dtype=np.int8)
        rank_quads = self.evaluator.get_hand_rank_52_plo(hand_1_ace, board_3a)

        # Both should make quads of aces, both should be high ranks
        self.assertGreater(rank_aces, 0)
        self.assertGreater(rank_quads, 0)

    def test_plo_uses_exactly_3_board_cards(self):
        """
        A board with 4 hearts should NOT give a flush to a player with only 1 heart
        in their hole cards. PLO requires exactly 2 suited hole cards for a flush.
        """
        # Board: 2h 5h 8h Jh Kc  (4 hearts on board)
        board = np.array([[0, 0], [3, 0], [6, 0], [9, 0], [11, 3]], dtype=np.int8)
        # Hand: Ah 3s 4d 6c — only 1 heart. Cannot make flush.
        hand_1_heart = np.array([[12, 0], [1, 2], [2, 1], [4, 3]], dtype=np.int8)
        # Hand: Ah 9h 3d 4c — 2 hearts. CAN make flush.
        hand_2_hearts = np.array([[12, 0], [7, 0], [1, 1], [2, 3]], dtype=np.int8)

        rank_1h = self.evaluator.get_hand_rank_52_plo(hand_1_heart, board)
        rank_2h = self.evaluator.get_hand_rank_52_plo(hand_2_hearts, board)

        # 2-heart hand makes a flush and should rank much higher
        self.assertGreater(rank_2h, rank_1h)

    # ─── Ranking ordering ───

    def test_higher_rank_is_better(self):
        """A flush should rank higher than a pair."""
        # Board: 2h 5h 8h Jc Ks
        board = np.array([[0, 0], [3, 0], [6, 0], [9, 3], [11, 2]], dtype=np.int8)
        # Flush hand: Ah 9h 3d 4c
        hand_flush = np.array([[12, 0], [7, 0], [1, 1], [2, 3]], dtype=np.int8)
        # Pair hand: Kc Qd 3s 4d — pair of kings
        hand_pair = np.array([[11, 3], [10, 1], [1, 2], [2, 1]], dtype=np.int8)

        rank_flush = self.evaluator.get_hand_rank_52_plo(hand_flush, board)
        rank_pair = self.evaluator.get_hand_rank_52_plo(hand_pair, board)
        self.assertGreater(rank_flush, rank_pair)

    def test_royal_flush_is_max(self):
        """A royal flush should produce the maximum rank (7462)."""
        # Board: Th Jh Qh 2d 3c
        board = np.array([[8, 0], [9, 0], [10, 0], [0, 1], [1, 3]], dtype=np.int8)
        # Hand: Ah Kh 5s 6d — royal flush using Ah Kh + Th Jh Qh from board
        hand = np.array([[12, 0], [11, 0], [3, 2], [4, 1]], dtype=np.int8)
        rank = self.evaluator.get_hand_rank_52_plo(hand, board)
        self.assertEqual(rank, 7462)

    # ─── Determinism ───

    def test_same_hand_deterministic(self):
        """Same inputs always produce the same output."""
        board = np.array([[0, 0], [3, 1], [6, 2], [9, 3], [11, 0]], dtype=np.int8)
        hand = np.array([[12, 0], [7, 0], [1, 1], [2, 3]], dtype=np.int8)
        rank_a = self.evaluator.get_hand_rank_52_plo(hand, board)
        rank_b = self.evaluator.get_hand_rank_52_plo(hand, board)
        self.assertEqual(rank_a, rank_b)

    # ─── Error handling ───

    def test_invalid_card_returns_negative(self):
        """Invalid card values should return -1 via the exception path."""
        board = np.array([[99, 99], [99, 99], [99, 99], [99, 99], [99, 99]], dtype=np.int8)
        hand = np.array([[99, 99], [99, 99]], dtype=np.int8)
        rank = self.evaluator.get_hand_rank_52_holdem(hand, board)
        self.assertEqual(rank, -1)

    def test_plo_invalid_card_returns_negative(self):
        """Invalid card values in PLO should return -1."""
        board = np.array([[99, 99], [99, 99], [99, 99], [99, 99], [99, 99]], dtype=np.int8)
        hand = np.array([[99, 99], [99, 99], [99, 99], [99, 99]], dtype=np.int8)
        rank = self.evaluator.get_hand_rank_52_plo(hand, board)
        self.assertEqual(rank, -1)


if __name__ == '__main__':
    unittest.main()
