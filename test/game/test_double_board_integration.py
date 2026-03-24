"""
Integration tests for Double Board Bomb Pot: environment + strategy interaction,
state round-trips, deck sync, and legal action consistency.
"""

import unittest
import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.games import DoubleBoardBombPotPLO


def _get_new_dbbp_env(n_seats=2):
    args = DoubleBoardBombPotPLO.ARGS_CLS(
        n_seats=n_seats,
        bet_sizes_list_as_frac_of_pot=[0.5, 1.0],
        stack_randomization_range=(0, 0),
        starting_stack_sizes_list=[10000] * n_seats)
    lut = DoubleBoardBombPotPLO.get_lut_holder()
    return DoubleBoardBombPotPLO(env_args=args, lut_holder=lut, is_evaluating=True)


class TestObservationConsistency(unittest.TestCase):

    def test_obs_shape_consistent(self):
        """Multiple resets should produce the same observation shape."""
        env = _get_new_dbbp_env()
        shapes = set()
        for _ in range(20):
            obs, _, _, _ = env.reset()
            shapes.add(obs.shape)
        self.assertEqual(len(shapes), 1, f"Inconsistent obs shapes: {shapes}")

    def test_obs_terminal_is_zeros(self):
        """Terminal observation should be all zeros."""
        env = _get_new_dbbp_env()
        env.reset()
        terminal = False
        while not terminal:
            obs, _, terminal, _ = env.step(env.get_random_action())
        # obs at terminal should be all zeros
        np.testing.assert_array_equal(obs, np.zeros_like(obs))


class TestStateDictRoundTrip(unittest.TestCase):

    def test_state_dict_round_trip(self):
        """Save → act → restore → obs should match saved obs."""
        env = _get_new_dbbp_env()
        for _ in range(30):
            obs_reset, _, terminal, _ = env.reset()
            if terminal:
                continue

            # Take a few actions
            steps = 0
            while not terminal and steps < 3:
                obs_reset, _, terminal, _ = env.step(env.get_random_action())
                steps += 1
            if terminal:
                continue

            # Save state
            saved_state = env.state_dict()
            saved_obs = env.get_current_obs(is_terminal=False)

            # Take more actions
            steps = 0
            while not terminal and steps < 3:
                _, _, terminal, _ = env.step(env.get_random_action())
                steps += 1

            # Restore
            env.load_state_dict(saved_state)
            restored_obs = env.get_current_obs(is_terminal=False)
            np.testing.assert_array_equal(saved_obs, restored_obs)


class TestDeckSync(unittest.TestCase):

    def test_deck_sync_between_envs(self):
        """Two DBBP envs with synced decks should draw identical cards."""
        env1 = _get_new_dbbp_env()
        env2 = _get_new_dbbp_env()

        env1.reset()
        env2.reset()
        env2.load_cards_state_dict(cards_state_dict=env1.cards_state_dict())

        # Both should have same board and hole cards after sync
        np.testing.assert_array_equal(env1.board, env2.board)
        for i in range(env1.N_SEATS):
            np.testing.assert_array_equal(env1.seats[i].hand, env2.seats[i].hand)


class TestLegalActions(unittest.TestCase):

    def test_legal_actions_always_nonempty(self):
        """Legal actions should never be empty during a game."""
        env = _get_new_dbbp_env()
        for _ in range(100):
            env.reset()
            terminal = False
            while not terminal:
                legal = env.get_legal_actions()
                self.assertGreater(len(legal), 0, "Legal actions list is empty!")
                _, _, terminal, _ = env.step(env.get_random_action())

    def test_fold_legal_when_facing_bet(self):
        """After a raise, FOLD should be a legal action for the next player."""
        env = _get_new_dbbp_env()
        for _ in range(50):
            env.reset()
            terminal = False
            # Try to find a state where someone has raised
            while not terminal:
                legal = env.get_legal_actions()
                if Poker.BET_RAISE in legal:
                    # Make a raise (action index 2 or higher in discretized env)
                    raise_actions = [a for a in legal if a >= 2]
                    if raise_actions:
                        _, _, terminal, _ = env.step(raise_actions[0])
                        if not terminal:
                            next_legal = env.get_legal_actions()
                            self.assertIn(Poker.FOLD, next_legal,
                                          "FOLD should be legal when facing a bet")
                        break
                _, _, terminal, _ = env.step(env.get_random_action())

    def test_check_call_always_available(self):
        """CHECK/CALL should always be a legal action (can always call or check)."""
        env = _get_new_dbbp_env()
        for _ in range(50):
            env.reset()
            terminal = False
            while not terminal:
                legal = env.get_legal_actions()
                self.assertIn(Poker.CHECK_CALL, legal,
                              "CHECK/CALL should always be legal")
                _, _, terminal, _ = env.step(env.get_random_action())


class TestMultipleGamesSequential(unittest.TestCase):
    """Verify environment is properly reset between games."""

    def test_sequential_games_independent(self):
        """Playing 100 sequential games should work cleanly."""
        env = _get_new_dbbp_env()
        for i in range(100):
            env.reset()
            terminal = False
            step_count = 0
            while not terminal:
                _, _, terminal, _ = env.step(env.get_random_action())
                step_count += 1
                # Safety: bomb pot games shouldn't take more than 100 steps
                self.assertLess(step_count, 200,
                                f"Game {i} seems stuck after {step_count} steps")


if __name__ == '__main__':
    unittest.main()
