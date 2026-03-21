"""
Pure Python replacement for CppLibHoldemLuts.
Provides card conversion and LUT generation without C shared libraries.
"""

import itertools
import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import HoldemRules


N_SUITS = 4
N_RANKS = 13
N_CARDS = N_RANKS * N_SUITS


def _get_1d_card(rank, suit):
    """Convert (rank, suit) to 1D card index."""
    return rank * N_SUITS + suit


def _get_2d_card(card_1d):
    """Convert 1D card index to (rank, suit)."""
    return np.array([card_1d // N_SUITS, card_1d % N_SUITS], dtype=np.int8)


class PythonLibHoldemLuts:
    """
    Pure Python replacement for CppLibHoldemLuts.
    Provides the same interface for card/LUT operations.
    """

    def __init__(self, n_boards_lut, n_cards_out_lut):
        self._n_boards_lut = n_boards_lut
        self._n_cards_out_lut = n_cards_out_lut

    def get_idx_2_hole_card_lut(self):
        """Generate LUT: range_idx -> (card1_1d, card2_1d) for holdem."""
        lut = np.full(shape=(HoldemRules.RANGE_SIZE, 2), fill_value=-2, dtype=np.int8)
        idx = 0
        for c1 in range(N_CARDS):
            for c2 in range(c1 + 1, N_CARDS):
                lut[idx, 0] = c1
                lut[idx, 1] = c2
                idx += 1
        return lut

    def get_hole_card_2_idx_lut(self):
        """Generate LUT: (card1_1d, card2_1d) -> range_idx for holdem."""
        lut = np.full(shape=(N_CARDS, N_CARDS), fill_value=-2, dtype=np.int16)
        idx = 0
        for c1 in range(N_CARDS):
            for c2 in range(c1 + 1, N_CARDS):
                lut[c1, c2] = idx
                idx += 1
        return lut

    def get_idx_2_flop_lut(self):
        n_boards = self._n_boards_lut[Poker.FLOP]
        n_cards_out = self._n_cards_out_lut[Poker.FLOP]
        lut = np.full(shape=(n_boards, n_cards_out), fill_value=-2, dtype=np.int8)
        for idx, combo in enumerate(itertools.combinations(range(N_CARDS), n_cards_out)):
            lut[idx] = combo
        return lut

    def get_idx_2_turn_lut(self):
        n_boards = self._n_boards_lut[Poker.TURN]
        n_cards_out = self._n_cards_out_lut[Poker.TURN]
        lut = np.full(shape=(n_boards, n_cards_out), fill_value=-2, dtype=np.int8)
        for idx, combo in enumerate(itertools.combinations(range(N_CARDS), n_cards_out)):
            lut[idx] = combo
        return lut

    def get_idx_2_river_lut(self):
        n_boards = self._n_boards_lut[Poker.RIVER]
        n_cards_out = self._n_cards_out_lut[Poker.RIVER]
        lut = np.full(shape=(n_boards, n_cards_out), fill_value=-2, dtype=np.int8)
        for idx, combo in enumerate(itertools.combinations(range(N_CARDS), n_cards_out)):
            lut[idx] = combo
        return lut

    def get_1d_card(self, card_2d):
        return _get_1d_card(card_2d[0], card_2d[1])

    def get_2d_card(self, card_1d):
        return _get_2d_card(card_1d)
