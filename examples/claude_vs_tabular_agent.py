"""
Claude's strategy plays against a trained CFR agent.
Supports both tree-based CFR and ES-MCCFR strategies.

Usage:
    # ES-MCCFR strategy (default):
    python examples/claude_vs_tabular_agent.py --algo es-mccfr --strategy ./strategies/DBBP_PLO_50b_10000iter_plus_esmccfr.npz --n-buckets 50

    # Tree-based CFR strategy:
    python examples/claude_vs_tabular_agent.py --algo tree-cfr --strategy ./strategies/DBBP_PLO_200b_1000iter_plus.npz --n-buckets 200
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


# ──────────────────────────────────────────────────────────────────────────────
# ES-MCCFR Agent
# ──────────────────────────────────────────────────────────────────────────────

class ESMCCFRAgent:
    """
    Plays using a saved ES-MCCFR strategy.
    Tracks (round, action_history) to look up info sets.
    Uses flop equity buckets + rank-based turn/river bucketing.
    Supports 2D bucketing for DBBP (separate bucket per board).
    """

    def __init__(self, strategy_path, n_buckets=50, n_rollouts=10,
                 cache_dir='./abstraction_cache', bet_set=None,
                 n_buckets_per_board=None):
        print("Building ES-MCCFR agent...")

        self._n_buckets = n_buckets
        game_cls = DoubleBoardBombPotPLO
        self._is_dbbp = getattr(game_cls, 'N_BOARDS', 1) > 1

        if n_buckets_per_board is not None:
            self._n_buckets_per_board = n_buckets_per_board
        else:
            self._n_buckets_per_board = int(np.sqrt(n_buckets)) if self._is_dbbp else n_buckets

        if bet_set is None:
            bet_set = bet_sets.PL_2

        self._env_args = game_cls.ARGS_CLS(
            n_seats=2,
            starting_stack_sizes_list=[game_cls.DEFAULT_STACK_SIZE, game_cls.DEFAULT_STACK_SIZE],
            bet_sizes_list_as_frac_of_pot=bet_set,
        )
        self._env_bldr = HistoryEnvBuilder(
            env_cls=get_env_cls_from_str(game_cls.__name__),
            env_args=self._env_args,
        )
        self._lut = self._env_bldr.lut_holder
        self._n_actions = self._env_args.N_ACTIONS

        self._card_abs = CardAbstraction(
            rules=self._env_bldr.rules,
            lut_holder=self._lut,
            n_buckets=n_buckets,
            n_rollouts=n_rollouts,
            cache_dir=cache_dir,
            n_buckets_per_board=self._n_buckets_per_board if self._is_dbbp else None,
        )

        # Precompute flop buckets
        print("  Loading flop buckets...")
        temp_env = self._env_bldr.get_new_env(is_evaluating=True)
        temp_env.reset()
        flop_board = np.copy(temp_env.board)
        self._flop_buckets = self._card_abs.get_postflop_buckets(flop_board)

        # Precompute rank boundaries for turn/river bucketing
        print("  Computing rank boundaries...")
        if self._is_dbbp:
            self._rank_boundaries_b1, self._rank_boundaries_b2 = \
                self._compute_rank_boundaries_2d(flop_board)
        else:
            self._rank_boundaries = self._compute_rank_boundaries(flop_board)

        # Load strategy — supports both strategy files and checkpoint files
        print("  Loading strategy from {}...".format(strategy_path))
        self._strategy = {}
        data = np.load(strategy_path, allow_pickle=True)

        if 'n_info_sets' in data.files:
            # Checkpoint format: key_0, strat_0, regret_0, ...
            import ast
            n = int(data['n_info_sets'][0])
            for i in range(n):
                key = ast.literal_eval(str(data['key_{}'.format(i)][0]))
                strat = data['strat_{}'.format(i)].astype(np.float64)
                row_sums = strat.sum(axis=1, keepdims=True)
                uniform = np.ones_like(strat) / max(strat.shape[1], 1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._strategy[key] = np.where(row_sums > 0, strat / row_sums, uniform)
        else:
            # Strategy format: repr(key) -> strategy array
            for str_key in data.files:
                key = eval(str_key)
                strat = data[str_key]
                row_sums = strat.sum(axis=1, keepdims=True)
                uniform = np.ones_like(strat) / max(strat.shape[1], 1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._strategy[key] = np.where(row_sums > 0, strat / row_sums, uniform)

        print("  Loaded {} info sets.".format(len(self._strategy)))
        print("  Agent ready.")

        self._action_history = ()
        self._current_round = None

    def reset(self):
        self._action_history = ()
        self._current_round = None

    def set_round(self, current_round):
        self._current_round = current_round

    def get_action(self, hand_2d, board, legal_actions, current_round):
        """Choose action from the ES-MCCFR average strategy."""
        range_idx = self._lut.get_range_idx_from_hole_cards(hand_2d)
        bucket = self._get_bucket(current_round, range_idx, board)

        info_key = (current_round, self._action_history)

        if info_key in self._strategy:
            strat_row = self._strategy[info_key][bucket]
            # Map strategy indices to legal actions
            action_probs = np.zeros(len(legal_actions), dtype=np.float64)
            for i, a in enumerate(legal_actions):
                if a < len(strat_row):
                    action_probs[i] = max(strat_row[a], 0)
            total = action_probs.sum()
            if total > 0:
                action_probs /= total
            else:
                action_probs = np.ones(len(legal_actions)) / len(legal_actions)
            chosen_idx = np.random.choice(len(legal_actions), p=action_probs)
            return legal_actions[chosen_idx]
        else:
            # Info set not seen during training — play uniform random
            return np.random.choice(legal_actions)

    def advance(self, action):
        self._action_history = self._action_history + (action,)

    def _get_bucket(self, current_round, range_idx, board):
        if current_round == Poker.FLOP:
            b = int(self._flop_buckets[range_idx])
            return max(b, 0)

        if self._is_dbbp:
            if current_round == Poker.TURN:
                eqs = self._eval_equity_per_board_exhaustive(range_idx, board)
            else:
                eqs = self._eval_hand_per_board(range_idx, board)
            npb = self._n_buckets_per_board
            b1 = int(np.searchsorted(self._rank_boundaries_b1, eqs[0]))
            b1 = min(b1, npb - 1)
            b2 = int(np.searchsorted(self._rank_boundaries_b2, eqs[1]))
            b2 = min(b2, npb - 1)
            return b1 * npb + b2
        else:
            if current_round == Poker.TURN:
                rank = self._eval_equity_avg_exhaustive(range_idx, board)
            else:
                rank = self._eval_hand_on_boards(range_idx, board)
            b = int(np.searchsorted(self._rank_boundaries, rank))
            return min(b, self._n_buckets - 1)

    def _eval_equity_per_board_exhaustive(self, range_idx, board):
        """
        On the turn, run out every possible river card and compute average
        hand rank per board. ~30 unknown cards = exact equity.
        """
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS[range_idx]
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD
        hand_2d = np.ascontiguousarray(
            np.array([card_2d_lut[c] for c in hole_cards], dtype=np.int8)
        )

        known_1d = set(hole_cards.tolist())
        for c in board:
            if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                known_1d.add(int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))
        remaining = [c for c in range(52) if c not in known_1d]

        total_slots = len(board)
        board_infos = []
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            dealt = [c for c in board[start:end] if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D]
            board_infos.append((dealt, len(dealt)))

        avg_ranks = []
        for dealt, n_dealt in board_infos:
            if n_dealt < 3:
                avg_ranks.append(0)
                continue
            if n_dealt == 5:
                board_5 = np.ascontiguousarray(np.array(dealt, dtype=np.int8))
                avg_ranks.append(hand_evaluator.get_hand_rank_52_plo(hand_2d, board_5))
                continue

            # n_dealt == 4: run out all river cards
            board_5 = np.empty((5, 2), dtype=np.int8)
            for i, c in enumerate(dealt):
                board_5[i] = c

            rank_sum = 0
            count = 0
            for rc in remaining:
                board_5[4] = card_2d_lut[rc]
                board_5_c = np.ascontiguousarray(board_5)
                rank_sum += hand_evaluator.get_hand_rank_52_plo(hand_2d, board_5_c)
                count += 1
            avg_ranks.append(rank_sum / count if count > 0 else 0)

        return avg_ranks

    def _eval_equity_avg_exhaustive(self, range_idx, board):
        """Average exhaustive equity across all boards."""
        eqs = self._eval_equity_per_board_exhaustive(range_idx, board)
        if not eqs:
            return 0
        return sum(eqs) / len(eqs)

    def _eval_hand_per_board(self, range_idx, board):
        """Evaluate hand rank on each board separately. Returns list of ranks."""
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS[range_idx]
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD
        hand_2d = np.ascontiguousarray(
            np.array([card_2d_lut[c] for c in hole_cards], dtype=np.int8)
        )
        total_slots = len(board)
        ranks = []
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            board_slice = board[start:end]
            dealt = [c for c in board_slice if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D]
            if len(dealt) < 3:
                ranks.append(0)
                continue
            board_5 = np.empty((5, 2), dtype=np.int8)
            for i, c in enumerate(dealt):
                board_5[i] = c
            if len(dealt) < 5:
                hand_1d = set(hole_cards.tolist())
                board_1d = set()
                for c in dealt:
                    board_1d.add(int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))
                pad_idx = len(dealt)
                for pad_c in range(52):
                    if pad_c not in hand_1d and pad_c not in board_1d:
                        board_5[pad_idx] = card_2d_lut[pad_c]
                        board_1d.add(pad_c)
                        pad_idx += 1
                        if pad_idx >= 5:
                            break
            board_5 = np.ascontiguousarray(board_5)
            rank = hand_evaluator.get_hand_rank_52_plo(hand_2d, board_5)
            ranks.append(rank)
        return ranks

    def _eval_hand_on_boards(self, range_idx, board):
        """Average rank across all boards (legacy 1D bucketing)."""
        ranks = self._eval_hand_per_board(range_idx, board)
        if not ranks:
            return 0
        return sum(ranks) / len(ranks)

    def _compute_rank_boundaries(self, flop_board):
        valid_mask = self._card_abs.get_blocked_mask(flop_board)
        valid_indices = np.where(valid_mask)[0]
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD

        boards_5 = []
        total_slots = len(flop_board)
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            dealt = np.array(
                [c for c in flop_board[start:end] if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D],
                dtype=np.int8
            )
            if len(dealt) > 0:
                if len(dealt) < 5:
                    board_1d = set()
                    for c in dealt:
                        board_1d.add(int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))
                    padded = np.empty((5, 2), dtype=np.int8)
                    padded[:len(dealt)] = dealt
                    pad_idx = len(dealt)
                    for pc in range(52):
                        if pc not in board_1d:
                            padded[pad_idx] = card_2d_lut[pc]
                            board_1d.add(pc)
                            pad_idx += 1
                            if pad_idx >= 5:
                                break
                    dealt = np.ascontiguousarray(padded)
                boards_5.append(dealt)

        sample_size = min(5000, len(valid_indices))
        sample_idx = np.random.choice(valid_indices, size=sample_size, replace=False)
        ranks = []
        for ridx in sample_idx:
            hand_2d = np.ascontiguousarray(
                np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
            )
            avg_rank = 0.0
            for board in boards_5:
                avg_rank += hand_evaluator.get_hand_rank_52_plo(hand_2d, board)
            avg_rank /= len(boards_5)
            ranks.append(avg_rank)

        ranks = np.array(ranks)
        percentiles = np.linspace(0, 100, self._n_buckets + 1)[1:-1]
        return np.percentile(ranks, percentiles)

    def _compute_rank_boundaries_2d(self, flop_board):
        """Compute separate rank boundaries per board for 2D bucketing."""
        valid_mask = self._card_abs.get_blocked_mask(flop_board)
        valid_indices = np.where(valid_mask)[0]
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD

        boards_5 = []
        total_slots = len(flop_board)
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            dealt = np.array(
                [c for c in flop_board[start:end] if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D],
                dtype=np.int8
            )
            if len(dealt) > 0:
                if len(dealt) < 5:
                    board_1d = set()
                    for c in dealt:
                        board_1d.add(int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))
                    padded = np.empty((5, 2), dtype=np.int8)
                    padded[:len(dealt)] = dealt
                    pad_idx = len(dealt)
                    for pc in range(52):
                        if pc not in board_1d:
                            padded[pad_idx] = card_2d_lut[pc]
                            board_1d.add(pc)
                            pad_idx += 1
                            if pad_idx >= 5:
                                break
                    dealt = np.ascontiguousarray(padded)
                boards_5.append(dealt)

        sample_size = min(5000, len(valid_indices))
        sample_idx = np.random.choice(valid_indices, size=sample_size, replace=False)

        ranks_per_board = [[] for _ in boards_5]
        for ridx in sample_idx:
            hand_2d = np.ascontiguousarray(
                np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
            )
            for bi, board in enumerate(boards_5):
                rank = hand_evaluator.get_hand_rank_52_plo(hand_2d, board)
                ranks_per_board[bi].append(rank)

        npb = self._n_buckets_per_board
        percentiles = np.linspace(0, 100, npb + 1)[1:-1]
        boundaries = []
        for bi in range(len(boards_5)):
            r = np.array(ranks_per_board[bi])
            boundaries.append(np.percentile(r, percentiles))

        if len(boundaries) >= 2:
            return boundaries[0], boundaries[1]
        return boundaries[0], boundaries[0]


def play_match(strategy_path, n_hands=100000, report_interval=1000,
               n_buckets=200, algo='es-mccfr', cache_dir='./abstraction_cache',
               n_rollouts=10, n_buckets_per_board=None):

    use_es_mccfr = (algo == 'es-mccfr')

    if use_es_mccfr:
        agent = ESMCCFRAgent(strategy_path, n_buckets=n_buckets,
                             n_rollouts=n_rollouts, cache_dir=cache_dir,
                             n_buckets_per_board=n_buckets_per_board)
        bet_set = bet_sets.PL_2
    else:
        agent = TabularCFRAgent(strategy_path, n_buckets=n_buckets, cache_dir=cache_dir)
        bet_set = bet_sets.POT_ONLY

    game_cls = DoubleBoardBombPotPLO
    env_args = game_cls.ARGS_CLS(
        n_seats=2,
        starting_stack_sizes_list=[game_cls.DEFAULT_STACK_SIZE, game_cls.DEFAULT_STACK_SIZE],
        bet_sizes_list_as_frac_of_pot=bet_set,
    )
    env = game_cls(env_args=env_args, is_evaluating=True,
                   lut_holder=game_cls.get_lut_holder())

    algo_label = 'ES-MCCFR' if use_es_mccfr else 'Tree CFR'
    total_winnings = 0
    interval_winnings = 0
    stats = {'my_folds': 0, 'ai_folds': 0, 'my_bets': 0, 'ai_bets': 0,
             'my_checks': 0, 'ai_checks': 0, 'showdowns': 0, 'scoops_me': 0,
             'scoops_ai': 0, 'splits': 0}

    print("\n" + "=" * 80)
    print("  CLAUDE STRATEGY vs {} ({} buckets)  |  {:,} hands".format(
        algo_label, n_buckets, n_hands))
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
        if use_es_mccfr:
            agent.set_round(env.current_round)
        else:
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
                if use_es_mccfr:
                    action = agent.get_action(ai_hand, env.board, legal, env.current_round)
                else:
                    action = agent.get_action(ai_hand, legal)
                if action == 0: stats['ai_folds'] += 1
                elif action == 1: stats['ai_checks'] += 1
                else: stats['ai_bets'] += 1

            action_tuple = env._get_env_adjusted_action_formulation(action)
            action_tuple = env._get_fixed_action(action_tuple)
            obs, rews, done, info = env._step(processed_action=action_tuple)

            agent.advance(action)

            if not done and env.current_round != last_round:
                if use_es_mccfr:
                    agent.set_round(env.current_round)
                else:
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
    parser = argparse.ArgumentParser(description='Claude vs CFR Agent')
    parser.add_argument('--algo', choices=['es-mccfr', 'tree-cfr'], default='es-mccfr',
                        help='Algorithm used for training (default: es-mccfr)')
    parser.add_argument('--strategy', type=str, required=True,
                        help='Path to saved .npz strategy file')
    parser.add_argument('--n-hands', type=int, default=100000,
                        help='Number of hands (default: 100000)')
    parser.add_argument('--interval', type=int, default=1000,
                        help='Report every N hands (default: 1000)')
    parser.add_argument('--n-buckets', type=int, default=225,
                        help='Number of buckets (must match training, default: 225)')
    parser.add_argument('--n-buckets-per-board', type=int, default=None,
                        help='Buckets per board for 2D bucketing (default: sqrt(n-buckets))')
    parser.add_argument('--rollouts', type=int, default=10,
                        help='MC rollouts for bucketing (must match training, default: 10)')
    parser.add_argument('--cache-dir', type=str, default='./abstraction_cache',
                        help='Abstraction cache directory (must match training)')
    args = parser.parse_args()

    play_match(
        strategy_path=args.strategy,
        n_hands=args.n_hands,
        report_interval=args.interval,
        n_buckets=args.n_buckets,
        algo=args.algo,
        cache_dir=args.cache_dir,
        n_rollouts=args.rollouts,
        n_buckets_per_board=args.n_buckets_per_board,
    )
