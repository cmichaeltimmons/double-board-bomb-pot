"""
Tests for the Double Board Bomb Pot environment: dealing, encoding, reset,
min raise, hand evaluation, and pot distribution (50/50 split).
"""

import unittest
import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.games import DoubleBoardBombPotPLO


def _get_new_dbbp_env(n_seats=2):
    """Create a fresh Double Board Bomb Pot environment."""
    args = DoubleBoardBombPotPLO.ARGS_CLS(
        n_seats=n_seats,
        bet_sizes_list_as_frac_of_pot=[0.5, 1.0],
        stack_randomization_range=(0, 0),
        starting_stack_sizes_list=[10000] * n_seats)
    lut = DoubleBoardBombPotPLO.get_lut_holder()
    return DoubleBoardBombPotPLO(env_args=args, lut_holder=lut, is_evaluating=True)


class TestResetAndInitialState(unittest.TestCase):
    """3A: Reset and initial state after bomb pot setup."""

    def test_reset_starts_on_flop(self):
        env = _get_new_dbbp_env()
        env.reset()
        self.assertEqual(env.current_round, Poker.FLOP)

    def test_reset_posts_antes(self):
        """After reset, each player's stack = 10000 - 300 = 9700, pot = 600."""
        env = _get_new_dbbp_env(n_seats=2)
        env.reset()
        for p in env.seats:
            self.assertEqual(p.stack, 10000 - DoubleBoardBombPotPLO.ANTE)
        self.assertEqual(env.main_pot, DoubleBoardBombPotPLO.ANTE * 2)

    def test_reset_posts_antes_multiway(self):
        """6-player: pot = 6 * 300 = 1800."""
        env = _get_new_dbbp_env(n_seats=6)
        env.reset()
        self.assertEqual(env.main_pot, DoubleBoardBombPotPLO.ANTE * 6)

    def test_reset_deals_hole_cards(self):
        env = _get_new_dbbp_env()
        env.reset()
        for p in env.seats:
            self.assertEqual(p.hand.shape, (4, 2))
            # All cards should be valid (not CARD_NOT_DEALT_TOKEN_1D)
            for card in p.hand:
                self.assertNotEqual(card[0], Poker.CARD_NOT_DEALT_TOKEN_1D)

    def test_reset_deals_flop_both_boards(self):
        """Board 1 flop (0:3) and board 2 flop (5:8) dealt; turns/rivers undealt."""
        env = _get_new_dbbp_env()
        env.reset()
        # Board 1 flop
        for i in range(3):
            self.assertNotEqual(env.board[i][0], Poker.CARD_NOT_DEALT_TOKEN_1D,
                                f"Board 1 flop card {i} should be dealt")
        # Board 2 flop
        for i in range(5, 8):
            self.assertNotEqual(env.board[i][0], Poker.CARD_NOT_DEALT_TOKEN_1D,
                                f"Board 2 flop card {i} should be dealt")
        # Board 1 turn+river undealt
        for i in [3, 4]:
            self.assertEqual(env.board[i][0], Poker.CARD_NOT_DEALT_TOKEN_1D,
                             f"Board 1 slot {i} should be undealt")
        # Board 2 turn+river undealt
        for i in [8, 9]:
            self.assertEqual(env.board[i][0], Poker.CARD_NOT_DEALT_TOKEN_1D,
                             f"Board 2 slot {i} should be undealt")

    def test_fixed_flop_cards_match(self):
        """Fixed flop cards should exactly match the class defaults."""
        env = _get_new_dbbp_env()
        env.reset()
        expected_b1 = np.array([[11, 2], [5, 0], [0, 1]], dtype=np.int8)
        expected_b2 = np.array([[10, 0], [9, 2], [7, 3]], dtype=np.int8)
        np.testing.assert_array_equal(env.board[:3], expected_b1)
        np.testing.assert_array_equal(env.board[5:8], expected_b2)

    def test_no_duplicate_cards_dealt(self):
        """All dealt cards (hole + board) should be unique."""
        env = _get_new_dbbp_env()
        env.reset()
        all_cards = []
        # Hole cards
        for p in env.seats:
            for card in p.hand:
                all_cards.append(tuple(card))
        # Board cards (only dealt ones)
        for card in env.board:
            if card[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                all_cards.append(tuple(card))
        self.assertEqual(len(all_cards), len(set(all_cards)),
                         "Duplicate cards found in deal")


class TestBoardDealing(unittest.TestCase):
    """3B: Turn and river dealing fills correct slots."""

    def test_board_shape_10x2(self):
        env = _get_new_dbbp_env()
        env.reset()
        self.assertEqual(env.board.shape, (10, 2))

    def test_deal_turn_fills_correct_slots(self):
        env = _get_new_dbbp_env()
        env.reset()
        env._deal_turn()
        # Turn cards should be dealt
        self.assertNotEqual(env.board[3][0], Poker.CARD_NOT_DEALT_TOKEN_1D)
        self.assertNotEqual(env.board[8][0], Poker.CARD_NOT_DEALT_TOKEN_1D)
        # River cards should still be undealt
        self.assertEqual(env.board[4][0], Poker.CARD_NOT_DEALT_TOKEN_1D)
        self.assertEqual(env.board[9][0], Poker.CARD_NOT_DEALT_TOKEN_1D)

    def test_deal_river_fills_all_slots(self):
        env = _get_new_dbbp_env()
        env.reset()
        env._deal_turn()
        env._deal_river()
        for i in range(10):
            self.assertNotEqual(env.board[i][0], Poker.CARD_NOT_DEALT_TOKEN_1D,
                                f"Board slot {i} should be dealt after river")


class TestBoardStateEncoding(unittest.TestCase):
    """3C: Board state encoding handles the gap between boards correctly."""

    def test_encoding_length(self):
        """Encoding should be N_TOTAL_BOARD_CARDS * (N_RANKS + N_SUITS) = 10 * 17 = 170."""
        env = _get_new_dbbp_env()
        env.reset()
        encoding = env._get_board_state()
        self.assertEqual(len(encoding), 10 * (13 + 4))

    def test_encoding_board2_flop_nonzero(self):
        """
        After flop deal, board 2 flop (indices 5-7) should have nonzero encoding.
        This is the critical 'continue' vs 'break' fix test.
        """
        env = _get_new_dbbp_env()
        env.reset()
        encoding = env._get_board_state()
        K = 13 + 4  # N_RANKS + N_SUITS

        # Board 2 flop cards (indices 5, 6, 7) should have nonzero segments
        for card_idx in [5, 6, 7]:
            segment = encoding[card_idx * K: (card_idx + 1) * K]
            self.assertGreater(sum(segment), 0,
                               f"Board 2 flop card at index {card_idx} should have nonzero encoding")

    def test_encoding_undealt_slots_zero(self):
        """Undealt turn/river slots should have all-zero encoding segments."""
        env = _get_new_dbbp_env()
        env.reset()
        encoding = env._get_board_state()
        K = 13 + 4

        for card_idx in [3, 4, 8, 9]:
            segment = encoding[card_idx * K: (card_idx + 1) * K]
            self.assertEqual(sum(segment), 0,
                             f"Undealt slot {card_idx} should have zero encoding")


class TestMinRaise(unittest.TestCase):
    """3D: Min raise uses ANTE when BIG_BLIND is 0."""

    def test_min_raise_initial(self):
        """Initial min raise should use ANTE (300) as delta."""
        env = _get_new_dbbp_env()
        env.reset()
        min_raise = env._get_current_total_min_raise()
        # After antes are swept into pot, current bets are 0.
        # Min raise = 0 (max current bet) + 300 (ANTE delta) = 300
        self.assertGreaterEqual(min_raise, DoubleBoardBombPotPLO.ANTE)


class TestHandEvaluation(unittest.TestCase):
    """3E: Hand evaluation against two boards."""

    def test_both_board_ranks_set(self):
        """After a showdown (no one folded), both board ranks should be positive."""
        env = _get_new_dbbp_env()
        # Run until we find a hand that goes to showdown (no folds)
        found_showdown = False
        for _ in range(200):
            env.reset()
            terminal = False
            while not terminal:
                # Only check/call to force showdown
                _, _, terminal, _ = env.step(1)  # CHECK_CALL
            # Verify showdown happened (no one folded)
            if all(not p.folded_this_episode for p in env.seats):
                found_showdown = True
                for p in env.seats:
                    self.assertGreater(p.hand_rank_board1, 0,
                                       "hand_rank_board1 should be positive at showdown")
                    self.assertGreater(p.hand_rank_board2, 0,
                                       "hand_rank_board2 should be positive at showdown")
                break
        self.assertTrue(found_showdown, "Could not find a showdown hand in 200 attempts")

    def test_hand_rank_is_max_of_boards(self):
        """hand_rank should equal max(hand_rank_board1, hand_rank_board2)."""
        env = _get_new_dbbp_env()
        env.reset()
        terminal = False
        # Check/call to force showdown
        while not terminal:
            _, _, terminal, _ = env.step(1)  # CHECK_CALL
        for p in env.seats:
            if not p.folded_this_episode:
                self.assertEqual(p.hand_rank, max(p.hand_rank_board1, p.hand_rank_board2))


class TestPotDistributionHU(unittest.TestCase):
    """3F: Heads-up pot distribution with 50/50 board split."""

    def test_hu_winner_both_boards(self):
        """If P0 wins both boards, P0 gets the full pot."""
        env = _get_new_dbbp_env()
        env.reset()
        # Mock hand ranks
        env.seats[0].hand_rank_board1 = 7000
        env.seats[0].hand_rank_board2 = 7000
        env.seats[1].hand_rank_board1 = 1000
        env.seats[1].hand_rank_board2 = 1000
        env.seats[0].hand_rank = 7000
        env.seats[1].hand_rank = 1000

        pot = env.main_pot
        s0_before = env.seats[0].stack
        s1_before = env.seats[1].stack
        env._payout_pots_hu()

        self.assertEqual(env.seats[0].stack, s0_before + pot)
        self.assertEqual(env.seats[1].stack, s1_before)

    def test_hu_split_boards(self):
        """P0 wins board 1, P1 wins board 2 — each gets half."""
        env = _get_new_dbbp_env()
        env.reset()
        env.seats[0].hand_rank_board1 = 7000
        env.seats[0].hand_rank_board2 = 1000
        env.seats[1].hand_rank_board1 = 1000
        env.seats[1].hand_rank_board2 = 7000
        env.seats[0].hand_rank = 7000
        env.seats[1].hand_rank = 7000

        pot = env.main_pot
        half = pot / 2
        s0_before = env.seats[0].stack
        s1_before = env.seats[1].stack
        env._payout_pots_hu()

        self.assertEqual(env.seats[0].stack, s0_before + half)
        self.assertEqual(env.seats[1].stack, s1_before + half)

    def test_hu_tie_one_board(self):
        """P0 wins board 1, tie on board 2 — P0 gets 75%, P1 gets 25%."""
        env = _get_new_dbbp_env()
        env.reset()
        env.seats[0].hand_rank_board1 = 7000
        env.seats[0].hand_rank_board2 = 5000
        env.seats[1].hand_rank_board1 = 1000
        env.seats[1].hand_rank_board2 = 5000

        pot = env.main_pot
        half = pot / 2
        quarter = half / 2
        s0_before = env.seats[0].stack
        s1_before = env.seats[1].stack
        env._payout_pots_hu()

        self.assertAlmostEqual(env.seats[0].stack, s0_before + half + quarter)
        self.assertAlmostEqual(env.seats[1].stack, s1_before + quarter)

    def test_hu_tie_both_boards(self):
        """Tie on both boards — each player gets exactly half."""
        env = _get_new_dbbp_env()
        env.reset()
        env.seats[0].hand_rank_board1 = 5000
        env.seats[0].hand_rank_board2 = 5000
        env.seats[1].hand_rank_board1 = 5000
        env.seats[1].hand_rank_board2 = 5000

        pot = env.main_pot
        half = pot / 2
        s0_before = env.seats[0].stack
        s1_before = env.seats[1].stack
        env._payout_pots_hu()

        self.assertAlmostEqual(env.seats[0].stack, s0_before + half)
        self.assertAlmostEqual(env.seats[1].stack, s1_before + half)

    def test_hu_chip_conservation(self):
        """Sum of all stacks should be constant across 200 random hands."""
        env = _get_new_dbbp_env()
        original_sum = sum(p.stack for p in env.seats)

        for _ in range(200):
            env.reset()
            terminal = False
            while not terminal:
                _, _, terminal, _ = env.step(env.get_random_action())
            current_sum = sum(p.stack for p in env.seats)
            self.assertEqual(current_sum, original_sum,
                             f"Chip conservation violated: {current_sum} != {original_sum}")


class TestPotDistributionMulti(unittest.TestCase):
    """3G: Multi-player pot distribution."""

    def test_multi_chip_conservation_3p(self):
        """3-player chip conservation over 100 hands (strict — no tolerance)."""
        env = _get_new_dbbp_env(n_seats=3)
        original_sum = sum(p.stack for p in env.seats)
        for _ in range(100):
            env.reset()
            terminal = False
            while not terminal:
                _, _, terminal, _ = env.step(env.get_random_action())
            current_sum = sum(p.stack for p in env.seats)
            self.assertEqual(current_sum, original_sum)

    def test_multi_chip_conservation_6p(self):
        """6-player chip conservation over 50 hands."""
        env = _get_new_dbbp_env(n_seats=6)
        original_sum = sum(p.stack for p in env.seats)
        for _ in range(50):
            env.reset()
            terminal = False
            while not terminal:
                _, _, terminal, _ = env.step(env.get_random_action())
            current_sum = sum(p.stack for p in env.seats)
            self.assertEqual(current_sum, original_sum)

    def test_multi_fold_to_one(self):
        """When all but one fold, the remaining player gets the pot and chips are conserved."""
        env = _get_new_dbbp_env(n_seats=3)
        env.reset()
        # Capture sum AFTER reset (antes already posted, but chips just moved to pot)
        stack_sum = sum(p.stack for p in env.seats)
        pot_sum = env.main_pot + sum(env.side_pots)
        total_before = stack_sum + pot_sum

        # All players fold (action 0 = FOLD in discretized env)
        terminal = False
        while not terminal:
            _, _, terminal, _ = env.step(0)  # FOLD
        total_after = sum(p.stack for p in env.seats)
        self.assertEqual(total_after, total_before,
                         f"Chips lost after fold: {total_after} != {total_before}")


class TestFullGamePlaythrough(unittest.TestCase):
    """3H: Full random game playthroughs."""

    ITERATIONS = 200

    def test_random_games_complete(self):
        """200 random games should complete without exception."""
        env = _get_new_dbbp_env()
        for _ in range(self.ITERATIONS):
            env.reset()
            terminal = False
            while not terminal:
                _, _, terminal, _ = env.step(env.get_random_action())

    def test_reward_zero_before_terminal(self):
        """Non-terminal rewards should be zero arrays."""
        env = _get_new_dbbp_env()
        for _ in range(50):
            env.reset()
            terminal = False
            while not terminal:
                _, reward, terminal, _ = env.step(env.get_random_action())
                if not terminal:
                    np.testing.assert_array_equal(
                        reward, np.zeros(env.N_SEATS, dtype=np.int32))

    def test_rewards_sum_to_zero(self):
        """Terminal rewards should sum to 0 (zero-sum game)."""
        env = _get_new_dbbp_env()
        for _ in range(100):
            env.reset()
            terminal = False
            while not terminal:
                _, reward, terminal, _ = env.step(env.get_random_action())
            self.assertEqual(np.sum(reward), 0,
                             f"Rewards don't sum to zero: {reward}")


if __name__ == '__main__':
    unittest.main()
