"""
Claude's strategy plays against the trained tabular CFR agent.
Loads a saved .npz strategy from AbstractedCFR and plays DBBP PLO.

Usage:
    python examples/claude_vs_tabular_agent.py --strategy /workspace/strategies/DBBP_PLO_200b_1000iter_plus.npz
    python examples/claude_vs_tabular_agent.py --strategy strategies/DBBP_PLO_200b_1000iter_plus.npz --n-hands 50000
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game.games import DoubleBoardBombPotPLO
from PokerRL.game import bet_sets
from PokerRL.game.card_abstraction import CardAbstraction
from PokerRL.game._.tree.AbstractedPublicTree import AbstractedPublicTree
from PokerRL.game.wrappers import HistoryEnvBuilder
from PokerRL.rl.rl_util import get_env_cls_from_str

try:
    from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
    hand_evaluator = CppHandeval()
except Exception:
    from PokerRL.game._.cpp_wrappers.PythonHandeval import PythonHandeval
    hand_evaluator = PythonHandeval()

RANK_STR = '23456789TJQKA'
SUIT_SYMBOL = ['h', 'd', 's', 'c']


def card_str(card_2d):
    if card_2d[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
        return '--'
    return RANK_STR[card_2d[0]] + SUIT_SYMBOL[card_2d[1]]


def hand_str(cards):
    return ' '.join(card_str(c) for c in cards)


def get_hand_strength(hand_2d, board_2d):
    if board_2d[0][0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
        return -1
    dealt = [c for c in board_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D]
    if len(dealt) < 3:
        return -1
    board_5 = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    for i, c in enumerate(dealt):
        board_5[i] = c
    return hand_evaluator.get_hand_rank_52_plo(hand_2d, board_5)


def assess_both_boards(hand_2d, board):
    return get_hand_strength(hand_2d, board[0:5]), get_hand_strength(hand_2d, board[5:10])


def choose_action_claude(hand_2d, board, legal_actions, pot, my_stack, opp_stack, current_round):
    """Claude's simple hand-strength-based strategy (same as claude_vs_agent.py)."""
    r1, r2 = assess_both_boards(hand_2d, board)
    if r1 < 0 and r2 < 0:
        return 1 if 1 in legal_actions else 0

    def strength(r):
        if r < 0: return 0
        if r > 900000: return 3
        if r > 650000: return 2
        if r > 500000: return 1
        return 0

    s1, s2 = strength(r1), strength(r2)
    total = s1 + s2
    facing_bet = 0 in legal_actions
    stack_commitment = 1.0 - (my_stack / 10000.0) if my_stack < 10000 else 0.0

    if total >= 5:
        if 4 in legal_actions: return 4
        if 3 in legal_actions: return 3
        return max(legal_actions)
    elif total >= 4:
        if 3 in legal_actions: return 3
        if 2 in legal_actions: return 2
        return 1 if 1 in legal_actions else 0
    elif total >= 3:
        if not facing_bet:
            return 2 if 2 in legal_actions else 1
        else:
            return 1 if 1 in legal_actions else 0
    elif total >= 2:
        if not facing_bet:
            return 1
        return 1 if (1 in legal_actions and stack_commitment < 0.3) else 0
    elif total == 1:
        if not facing_bet:
            return 1
        return 1 if (1 in legal_actions and stack_commitment < 0.1) else 0
    else:
        return 1 if not facing_bet else 0


# ──────────────────────────────────────────────────────────────────────────────
# Tabular CFR Agent
# ──────────────────────────────────────────────────────────────────────────────

class TabularCFRAgent:
    """
    Plays using a saved tabular CFR strategy.
    Rebuilds the game tree and loads saved avg_strat arrays into nodes.
    At runtime, maps the current hand to a bucket and samples from the strategy.
    """

    def __init__(self, strategy_path, n_buckets=200, cache_dir='./abstraction_cache'):
        print("Building tabular CFR agent...")

        self._n_buckets = n_buckets
        game_cls = DoubleBoardBombPotPLO

        self._env_args = game_cls.ARGS_CLS(
            n_seats=2,
            starting_stack_sizes_list=[game_cls.DEFAULT_STACK_SIZE, game_cls.DEFAULT_STACK_SIZE],
            bet_sizes_list_as_frac_of_pot=bet_sets.POT_ONLY,
        )
        self._env_bldr = HistoryEnvBuilder(
            env_cls=get_env_cls_from_str(game_cls.__name__),
            env_args=self._env_args,
        )

        self._card_abs = CardAbstraction(
            rules=self._env_bldr.rules,
            lut_holder=self._env_bldr.lut_holder,
            n_buckets=n_buckets,
            n_rollouts=5000,
            cache_dir=cache_dir,
        )

        print("  Building game tree...")
        self._tree = AbstractedPublicTree(
            env_bldr=self._env_bldr,
            stack_size=[game_cls.DEFAULT_STACK_SIZE, game_cls.DEFAULT_STACK_SIZE],
            stop_at_street=None,
            card_abstraction=self._card_abs,
            n_buckets=n_buckets,
        )
        self._tree.build_tree()
        print("  Tree: {} nodes".format(self._tree.n_nodes))

        print("  Loading strategy from {}...".format(strategy_path))
        self._load_strategy(strategy_path)
        print("  Agent ready.")

        self._current_node = None

    def _load_strategy(self, path):
        """Load saved avg_strat arrays into tree nodes."""
        data = np.load(path)
        node_id = [0]

        def _fill(node):
            if node.is_terminal:
                return
            if node.p_id_acting_next != self._tree.CHANCE_ID:
                key = str(node_id[0])
                if key in data:
                    if node.data is None:
                        node.data = {}
                    node.data['avg_strat'] = data[key]
                    node.strategy = data[key].copy()
                else:
                    n_actions = len(node.children)
                    uniform = np.full((self._n_buckets, n_actions),
                                      1.0 / n_actions, dtype=np.float32)
                    if node.data is None:
                        node.data = {}
                    node.data['avg_strat'] = uniform
                    node.strategy = uniform.copy()
            node_id[0] += 1
            for c in node.children:
                _fill(c)

        _fill(self._tree.root)

    def reset(self):
        """Reset to root for a new hand."""
        self._current_node = self._tree.root

    def get_action(self, hand_2d, legal_actions):
        """Choose an action given the current hand and legal actions."""
        node = self._current_node
        if node is None or node.is_terminal:
            return np.random.choice(legal_actions)

        board_2d = node.env_state[EnvDictIdxs.board_2d]
        n_dealt = sum(1 for c in board_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)

        range_idx = self._env_bldr.lut_holder.get_range_idx_from_hole_cards(hand_2d)

        if n_dealt == 0:
            buckets = self._card_abs.get_preflop_buckets()
        else:
            buckets = self._card_abs.get_postflop_buckets(board_2d)
        bucket = buckets[range_idx]

        if bucket < 0:
            return np.random.choice(legal_actions)

        strategy = node.strategy[bucket]

        if len(strategy) != len(node.allowed_actions):
            return np.random.choice(legal_actions)

        # Build probability distribution over legal_actions
        action_probs = np.zeros(len(legal_actions), dtype=np.float64)
        for i, a in enumerate(legal_actions):
            if a in node.allowed_actions:
                tree_idx = node.allowed_actions.index(a)
                action_probs[i] = max(strategy[tree_idx], 0)

        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        else:
            action_probs = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)

        chosen_idx = np.random.choice(len(legal_actions), p=action_probs)
        return legal_actions[chosen_idx]

    def advance(self, action):
        """Advance the tree pointer after an action is taken."""
        node = self._current_node
        if node is None or node.is_terminal:
            return

        if node.p_id_acting_next == self._tree.CHANCE_ID:
            return

        for child in node.children:
            if hasattr(child, 'action') and child.action == action:
                self._current_node = child
                return

    def advance_to_board(self, board_2d):
        """After cards are dealt, find the matching chance child."""
        node = self._current_node
        if node is None or node.is_terminal:
            return
        if node.p_id_acting_next != self._tree.CHANCE_ID:
            return

        for child in node.children:
            child_board = child.env_state[EnvDictIdxs.board_2d]
            if np.array_equal(child_board, board_2d):
                self._current_node = child
                return


def play_match(strategy_path, n_hands=100000, report_interval=1000,
               n_buckets=200, cache_dir='./abstraction_cache'):

    agent = TabularCFRAgent(strategy_path, n_buckets=n_buckets, cache_dir=cache_dir)

    game_cls = DoubleBoardBombPotPLO
    env_args = game_cls.ARGS_CLS(
        n_seats=2,
        starting_stack_sizes_list=[game_cls.DEFAULT_STACK_SIZE, game_cls.DEFAULT_STACK_SIZE],
        bet_sizes_list_as_frac_of_pot=bet_sets.POT_ONLY,
    )
    env = game_cls(env_args=env_args, is_evaluating=True,
                   lut_holder=game_cls.get_lut_holder())

    total_winnings = 0
    interval_winnings = 0
    stats = {'my_folds': 0, 'ai_folds': 0, 'my_bets': 0, 'ai_bets': 0,
             'my_checks': 0, 'ai_checks': 0, 'showdowns': 0, 'scoops_me': 0,
             'scoops_ai': 0, 'splits': 0}

    print("\n" + "=" * 80)
    print("  CLAUDE STRATEGY vs TABULAR CFR  |  {:,} hands, report every {:,}".format(
        n_hands, report_interval))
    print("  Positions alternate each hand (even=P0, odd=P1)")
    print("=" * 80)
    print("  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>12}  {:>7}".format(
        'Hands', 'Interval', 'Total', 'mA/hand', 'Showdowns', 'Scoops(C/A)', 'Splits'))
    print("  {}  {}  {}  {}  {}  {}  {}".format(
        '-' * 8, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 12, '-' * 7))

    for hand_num in range(1, n_hands + 1):
        claude_seat = hand_num % 2
        agent_seat = 1 - claude_seat

        obs, _, done, _ = env.reset()
        agent.reset()
        agent.advance_to_board(env.board.copy())

        my_hand = env.seats[claude_seat].hand
        last_round = env.current_round

        while not done:
            p_id = env.current_player.seat_id
            legal = env.get_legal_actions()

            if p_id == claude_seat:
                action = choose_action_claude(
                    my_hand, env.board, legal, env.main_pot,
                    env.seats[claude_seat].stack, env.seats[agent_seat].stack,
                    env.current_round
                )
                if action == 0: stats['my_folds'] += 1
                elif action == 1: stats['my_checks'] += 1
                else: stats['my_bets'] += 1
            else:
                ai_hand = env.seats[agent_seat].hand
                action = agent.get_action(ai_hand, legal)
                if action == 0: stats['ai_folds'] += 1
                elif action == 1: stats['ai_checks'] += 1
                else: stats['ai_bets'] += 1

            action_tuple = env._get_env_adjusted_action_formulation(action)
            action_tuple = env._get_fixed_action(action_tuple)
            obs, rews, done, info = env._step(processed_action=action_tuple)

            agent.advance(action)

            if not done and env.current_round != last_round:
                agent.advance_to_board(env.board.copy())
                last_round = env.current_round

        reward = int(np.rint(rews[claude_seat] * env.REWARD_SCALAR))
        total_winnings += reward
        interval_winnings += reward

        if (hasattr(env.seats[claude_seat], 'hand_rank_board1') and
                env.seats[claude_seat].hand_rank_board1 > 0):
            stats['showdowns'] += 1
            b1_me = env.seats[claude_seat].hand_rank_board1
            b1_ai = env.seats[agent_seat].hand_rank_board1
            b2_me = env.seats[claude_seat].hand_rank_board2
            b2_ai = env.seats[agent_seat].hand_rank_board2
            if b1_me > b1_ai and b2_me > b2_ai: stats['scoops_me'] += 1
            elif b1_ai > b1_me and b2_ai > b2_me: stats['scoops_ai'] += 1
            else: stats['splits'] += 1

        if hand_num % report_interval == 0:
            ma_per_hand = (total_winnings / hand_num) * env.EV_NORMALIZER
            print("  {:>8,}  {:>+10,}  {:>+10,}  {:>+10.1f}  {:>10,}  {:>5,}/{:<5,}  {:>7,}".format(
                hand_num, interval_winnings, total_winnings, ma_per_hand,
                stats['showdowns'], stats['scoops_me'], stats['scoops_ai'], stats['splits']))
            interval_winnings = 0

    ma_per_hand = (total_winnings / n_hands) * env.EV_NORMALIZER
    print("\n" + "=" * 80)
    print("  FINAL RESULTS  |  {:,} hands".format(n_hands))
    print("=" * 80)
    print("  Total: {:+,} chips  |  {:+.1f} mA/hand".format(total_winnings, ma_per_hand))
    print("  Claude - bets: {:,}  checks: {:,}  folds: {:,}".format(
        stats['my_bets'], stats['my_checks'], stats['my_folds']))
    print("  Agent  - bets: {:,}  checks: {:,}  folds: {:,}".format(
        stats['ai_bets'], stats['ai_checks'], stats['ai_folds']))
    print("  Showdowns: {:,}  |  Claude scoops: {:,}  Agent scoops: {:,}  Splits: {:,}".format(
        stats['showdowns'], stats['scoops_me'], stats['scoops_ai'], stats['splits']))
    print("\n  {}".format(
        'CLAUDE STRATEGY WINS!' if total_winnings > 0 else
        'AGENT WINS!' if total_winnings < 0 else 'DRAW!'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Claude vs Tabular CFR Agent')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Path to saved .npz strategy file')
    parser.add_argument('--n-hands', type=int, default=100000,
                        help='Number of hands (default: 100000)')
    parser.add_argument('--interval', type=int, default=1000,
                        help='Report every N hands (default: 1000)')
    parser.add_argument('--n-buckets', type=int, default=200,
                        help='Number of buckets (must match training)')
    parser.add_argument('--cache-dir', type=str, default='./abstraction_cache',
                        help='Abstraction cache directory (must match training)')
    args = parser.parse_args()

    play_match(
        strategy_path=args.strategy,
        n_hands=args.n_hands,
        report_interval=args.interval,
        n_buckets=args.n_buckets,
        cache_dir=args.cache_dir,
    )
