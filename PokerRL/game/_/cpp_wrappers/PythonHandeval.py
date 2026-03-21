"""
Pure Python hand evaluator using the treys library.
Drop-in replacement for CppHandeval when C shared libraries aren't available (e.g. macOS ARM).
"""

import itertools
import numpy as np
from treys import Card, Evaluator


# Max treys rank (7462 = worst hand). We invert so higher = better to match CppHandeval convention.
_TREYS_MAX_RANK = 7462
_evaluator = Evaluator()


def _card_2d_to_treys(rank, suit):
    """Convert PokerRL 2D card (rank 0-12, suit 0-3) to treys Card int."""
    # PokerRL ranks: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
    # PokerRL suits: 0=h, 1=d, 2=s, 3=c
    rank_chars = '23456789TJQKA'
    suit_chars = 'hdsc'
    return Card.new(rank_chars[rank] + suit_chars[suit])


class PythonHandeval:
    """
    Pure Python replacement for CppHandeval.
    Provides the same interface: get_hand_rank_52_holdem and get_hand_rank_52_plo.
    """

    def get_hand_rank_52_holdem(self, hand_2d, board_2d):
        """
        Evaluate best 5-card hand from 2 hole cards + 5 board cards.

        Args:
            hand_2d: np.ndarray shape (2, 2) - [[rank, suit], [rank, suit]]
            board_2d: np.ndarray shape (5, 2) - board cards

        Returns:
            int: hand rank, higher is better
        """
        try:
            hand_treys = [_card_2d_to_treys(c[0], c[1]) for c in hand_2d]
            board_treys = [_card_2d_to_treys(c[0], c[1]) for c in board_2d]
            treys_rank = _evaluator.evaluate(hand_treys, board_treys)
            # Invert: treys uses 1=best, we need higher=better
            return _TREYS_MAX_RANK - treys_rank + 1
        except Exception:
            return -1

    def get_hand_rank_52_plo(self, hand_2d, board_2d):
        """
        Evaluate PLO hand: must use exactly 2 of 4 hole cards + 3 of 5 board cards.

        Args:
            hand_2d: np.ndarray shape (4, 2) - 4 hole cards
            board_2d: np.ndarray shape (5, 2) - 5 board cards

        Returns:
            int: best hand rank across all valid 2-hole + 3-board combinations, higher is better
        """
        maxres = -1
        # All C(4,2) = 6 combinations of 2 hole cards
        for h_idxs in itertools.combinations(range(4), 2):
            # All C(5,3) = 10 combinations of 3 board cards
            for b_idxs in itertools.combinations(range(5), 3):
                try:
                    hand_treys = [_card_2d_to_treys(hand_2d[i][0], hand_2d[i][1]) for i in h_idxs]
                    board_treys = [_card_2d_to_treys(board_2d[i][0], board_2d[i][1]) for i in b_idxs]
                    # Evaluate as 5-card hand (2 hole + 3 board)
                    five_cards = hand_treys + board_treys
                    treys_rank = _evaluator.evaluate([], five_cards)
                    rank = _TREYS_MAX_RANK - treys_rank + 1
                    if rank > maxres:
                        maxres = rank
                except Exception:
                    continue
        return maxres

    def get_hand_rank_all_hands_on_given_boards_52_holdem(self, boards_1d, lut_holder):
        """
        Compute hand rank for all possible holdem hands on given boards.
        This is used for LBR evaluation - not critical path for training.
        """
        from PokerRL.game._.rl_env.game_rules import HoldemRules
        n_boards = boards_1d.shape[0]
        hand_ranks = np.full(shape=(n_boards, HoldemRules.RANGE_SIZE), fill_value=-1, dtype=np.int32)

        for b_idx in range(n_boards):
            board_1d = boards_1d[b_idx]
            board_2d = lut_holder.LUT_1DCARD_2_2DCARD[board_1d]
            board_cards_set = set(board_1d.tolist())

            for range_idx in range(HoldemRules.RANGE_SIZE):
                hole_1d = lut_holder.LUT_IDX_2_HOLE_CARDS[range_idx]
                # Check for blocked cards
                if hole_1d[0] in board_cards_set or hole_1d[1] in board_cards_set:
                    continue
                hand_2d = lut_holder.LUT_1DCARD_2_2DCARD[hole_1d]
                hand_ranks[b_idx, range_idx] = self.get_hand_rank_52_holdem(hand_2d, board_2d)

        return hand_ranks
