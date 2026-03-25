"""
External Sampling MCCFR with card abstraction for PLO / DBBP.
No tree construction — samples one deal per iteration and traverses
depth-first using the actual environment.
"""

import ast
import json
import os
import time

import numpy as np

from PokerRL.game.card_abstraction import CardAbstraction
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import HistoryEnvBuilder
from PokerRL.rl.rl_util import get_env_cls_from_str


class ExternalSamplingMCCFR:

    def __init__(self,
                 name,
                 game_cls,
                 agent_bet_set,
                 n_buckets=50,
                 variant='plus',
                 starting_stack_sizes=None,
                 n_rollouts=500,
                 cache_dir='./abstraction_cache',
                 n_buckets_per_board=None):
        import sys
        self._name = name
        self._n_buckets = n_buckets
        self._variant = variant
        self._n_seats = 2
        self._is_dbbp = getattr(game_cls, 'N_BOARDS', 1) > 1

        # For DBBP 2D bucketing
        if n_buckets_per_board is not None:
            self._n_buckets_per_board = n_buckets_per_board
        else:
            self._n_buckets_per_board = int(np.sqrt(n_buckets)) if self._is_dbbp else n_buckets

        if starting_stack_sizes is None:
            starting_stack_sizes = [game_cls.DEFAULT_STACK_SIZE] * self._n_seats

        self._game_cls_str = game_cls.__name__

        # Build environment
        if getattr(game_cls, 'IS_FIXED_LIMIT_GAME', False):
            self._env_args = game_cls.ARGS_CLS(
                n_seats=self._n_seats,
                starting_stack_sizes_list=starting_stack_sizes,
            )
        else:
            self._env_args = game_cls.ARGS_CLS(
                n_seats=self._n_seats,
                starting_stack_sizes_list=starting_stack_sizes,
                bet_sizes_list_as_frac_of_pot=agent_bet_set,
            )
        self._env_bldr = HistoryEnvBuilder(
            env_cls=get_env_cls_from_str(self._game_cls_str),
            env_args=self._env_args,
        )

        # Card abstraction for bucket mapping
        self._card_abs = CardAbstraction(
            rules=self._env_bldr.rules,
            lut_holder=self._env_bldr.lut_holder,
            n_buckets=n_buckets,
            n_rollouts=n_rollouts,
            cache_dir=cache_dir,
            n_buckets_per_board=self._n_buckets_per_board if self._is_dbbp else None,
        )

        # Create environment instance
        self._env = self._env_bldr.get_new_env(is_evaluating=True)
        self._n_actions = self._env_args.N_ACTIONS

        # Info set map: (current_round, action_history) -> {regret_sum, strategy_sum}
        self._info_set_map = {}

        self._iter_counter = 0
        self._metrics = []

        # Precompute flop buckets (one-time, cached to disk).
        print("Precomputing flop buckets (one-time)...")
        sys.stdout.flush()
        self._env.reset()
        flop_board = np.copy(self._env.board)
        self._flop_buckets = self._card_abs.get_postflop_buckets(flop_board)
        print("Flop buckets ready.")
        sys.stdout.flush()

        # Precompute rank-based bucket boundaries for turn/river.
        print("Precomputing rank bucket boundaries...")
        sys.stdout.flush()
        self._lut = self._env_bldr.lut_holder
        self._hand_eval = self._card_abs._hand_eval
        if self._is_dbbp:
            self._rank_boundaries_b1, self._rank_boundaries_b2 = \
                self._compute_rank_boundaries_2d(flop_board)
            print("Rank boundaries ready (2D: {} per board).".format(
                len(self._rank_boundaries_b1) + 1))
        else:
            self._rank_boundaries = self._compute_rank_boundaries(flop_board)
            print("Rank boundaries ready ({} boundaries).".format(len(self._rank_boundaries)))
        sys.stdout.flush()

    @property
    def name(self):
        return self._name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iteration(self):
        """Run one ES-MCCFR iteration (traverse once per player role)."""
        for traverser in range(self._n_seats):
            self._env.reset()
            self._traverse(traverser, action_history=())

        self._iter_counter += 1

    def get_average_strategy(self):
        """Returns dict: info_key -> (n_buckets, n_actions) normalized strategy."""
        avg = {}
        for key, node in self._info_set_map.items():
            s = node['strategy_sum'].copy()
            row_sums = s.sum(axis=1, keepdims=True)
            uniform = np.ones_like(s) / max(self._n_actions, 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                avg[key] = np.where(row_sums > 0, s / row_sums, uniform)
        return avg

    def log_iteration(self, iteration, elapsed):
        n_info_sets = len(self._info_set_map)
        total_regret = sum(
            np.sum(np.abs(n['regret_sum'])) for n in self._info_set_map.values()
        )
        entry = {
            'iteration': iteration,
            'n_info_sets': n_info_sets,
            'total_abs_regret': float(total_regret),
            'time_seconds': elapsed,
        }
        self._metrics.append(entry)
        return entry

    def save_strategy(self, path):
        """Save average strategy and info set data."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        avg_strat = self.get_average_strategy()

        save_data = {}
        for key, strat in avg_strat.items():
            str_key = repr(key)
            save_data[str_key] = strat

        np.savez_compressed(path, **save_data)
        print("Saved strategy ({} info sets) to {}".format(len(save_data), path))

    def save_checkpoint(self, path):
        """Save full state for resuming training."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        arrays = {}
        for i, (key, node) in enumerate(self._info_set_map.items()):
            arrays['key_{}'.format(i)] = np.array([repr(key)])
            arrays['regret_{}'.format(i)] = node['regret_sum']
            arrays['strat_{}'.format(i)] = node['strategy_sum']
        arrays['n_info_sets'] = np.array([len(self._info_set_map)])
        arrays['metadata'] = np.array([self._iter_counter, self._n_buckets, self._n_actions])
        arrays['variant'] = np.array([self._variant])
        np.savez_compressed(path, **arrays)
        print("Saved checkpoint (iter {}, {} info sets) to {}".format(
            self._iter_counter, len(self._info_set_map), path))

    def load_checkpoint(self, path):
        """Load training state from checkpoint."""
        data = np.load(path, allow_pickle=True)
        self._iter_counter = int(data['metadata'][0])
        n = int(data['n_info_sets'][0])
        self._info_set_map = {}
        for i in range(n):
            key = ast.literal_eval(str(data['key_{}'.format(i)][0]))
            self._info_set_map[key] = {
                'regret_sum': data['regret_{}'.format(i)].astype(np.float64),
                'strategy_sum': data['strat_{}'.format(i)].astype(np.float64),
            }
        print("Loaded checkpoint: iter {}, {} info sets".format(
            self._iter_counter, len(self._info_set_map)))

    def save_metrics(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Core traversal
    # ------------------------------------------------------------------

    def _traverse(self, traverser, action_history):
        """
        External sampling MCCFR traversal.

        At traverser nodes: explore ALL legal actions, update regrets.
        At opponent nodes: sample ONE action from current strategy.
        Returns the expected value for the traverser.
        """
        acting_player = self._env.current_player.seat_id
        current_round = self._env.current_round
        legal_actions = self._env.get_legal_actions()

        # Get bucket for acting player
        range_idx = self._env.get_range_idx(acting_player)
        bucket = self._get_bucket(current_round, range_idx)

        # Get or create info set node
        info_key = (current_round, action_history)
        node = self._get_or_create_node(info_key)

        # Compute current strategy via regret matching
        strategy = self._regret_matching(node, legal_actions, bucket)

        if acting_player == traverser:
            return self._traverse_traverser(
                traverser, action_history, current_round,
                legal_actions, node, strategy, bucket
            )
        else:
            return self._traverse_opponent(
                traverser, action_history, current_round,
                legal_actions, strategy
            )

    def _traverse_traverser(self, traverser, action_history, current_round,
                            legal_actions, node, strategy, bucket):
        """Traverser explores ALL legal actions."""
        saved_state = self._env.state_dict()
        action_values = np.zeros(self._n_actions, dtype=np.float64)

        for a in legal_actions:
            self._env.load_state_dict(saved_state)
            obs, reward, done, info = self._env.step(a)

            if done:
                action_values[a] = float(reward[traverser])
            else:
                new_history = action_history + (a,)
                action_values[a] = self._traverse(traverser, new_history)

        # Node value under current strategy
        node_value = 0.0
        for a in legal_actions:
            node_value += strategy[a] * action_values[a]

        # Update regrets
        for a in legal_actions:
            regret = action_values[a] - node_value
            if self._variant == 'linear':
                regret *= (self._iter_counter + 1)
            node['regret_sum'][bucket, a] += regret

        # CFR+: clamp to non-negative
        if self._variant == 'plus':
            node['regret_sum'][bucket] = np.maximum(node['regret_sum'][bucket], 0)

        return node_value

    def _traverse_opponent(self, traverser, action_history, current_round,
                           legal_actions, strategy):
        """Opponent samples ONE action from strategy."""
        probs = np.array([strategy[a] for a in legal_actions], dtype=np.float64)
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
        else:
            probs = np.ones(len(legal_actions)) / len(legal_actions)

        sampled_idx = np.random.choice(len(legal_actions), p=probs)
        sampled_action = legal_actions[sampled_idx]

        obs, reward, done, info = self._env.step(sampled_action)

        if done:
            return float(reward[traverser])

        new_history = action_history + (sampled_action,)
        return self._traverse(traverser, new_history)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create_node(self, info_key):
        if info_key not in self._info_set_map:
            self._info_set_map[info_key] = {
                'regret_sum': np.zeros((self._n_buckets, self._n_actions), dtype=np.float64),
                'strategy_sum': np.zeros((self._n_buckets, self._n_actions), dtype=np.float64),
            }
        return self._info_set_map[info_key]

    def _regret_matching(self, node, legal_actions, bucket):
        """Compute strategy from regrets, accumulate into strategy_sum."""
        strategy = np.zeros(self._n_actions, dtype=np.float64)
        regrets = node['regret_sum'][bucket]

        pos_sum = 0.0
        for a in legal_actions:
            if regrets[a] > 0:
                pos_sum += regrets[a]

        if pos_sum > 0:
            for a in legal_actions:
                strategy[a] = max(regrets[a], 0) / pos_sum
        else:
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)

        # Accumulate for average strategy
        weight = (self._iter_counter + 1) if self._variant == 'linear' else 1.0
        node['strategy_sum'][bucket] += strategy * weight

        return strategy

    def _get_bucket(self, current_round, range_idx):
        """
        Per-street bucketing:
        - Flop: precomputed equity buckets (MC rollouts, cached)
        - Turn/River: instant rank-based buckets using current board
        For DBBP, uses 2D bucketing on turn/river.
        """
        if current_round == Poker.FLOP:
            b = int(self._flop_buckets[range_idx])
            return max(b, 0)

        board_2d = self._env.board

        if self._is_dbbp:
            # 2D bucketing: separate bucket per board
            ranks = self._eval_hand_per_board(range_idx, board_2d)
            npb = self._n_buckets_per_board
            b1 = int(np.searchsorted(self._rank_boundaries_b1, ranks[0]))
            b1 = min(b1, npb - 1)
            b2 = int(np.searchsorted(self._rank_boundaries_b2, ranks[1]))
            b2 = min(b2, npb - 1)
            return b1 * npb + b2
        else:
            rank = self._eval_hand_on_boards(range_idx, board_2d)
            b = int(np.searchsorted(self._rank_boundaries, rank))
            return min(b, self._n_buckets - 1)

    def _eval_hand_per_board(self, range_idx, board_2d):
        """Evaluate a single hand's rank on each board separately. Returns list of ranks."""
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS[range_idx]
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD
        hand_2d = np.ascontiguousarray(
            np.array([card_2d_lut[c] for c in hole_cards], dtype=np.int8)
        )

        total_slots = len(board_2d)
        ranks = []
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            board_slice = board_2d[start:end]
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
                for pad_c in range(52):
                    if pad_c not in hand_1d and pad_c not in board_1d:
                        board_5[len(dealt)] = card_2d_lut[pad_c]
                        if len(dealt) + 1 < 5:
                            board_1d.add(pad_c)
                            for pad_c2 in range(pad_c + 1, 52):
                                if pad_c2 not in hand_1d and pad_c2 not in board_1d:
                                    board_5[len(dealt) + 1] = card_2d_lut[pad_c2]
                                    break
                        break
            board_5 = np.ascontiguousarray(board_5)
            rank = self._hand_eval.get_hand_rank_52_plo(hand_2d, board_5)
            ranks.append(rank)

        return ranks

    def _eval_hand_on_boards(self, range_idx, board_2d):
        """Evaluate a single hand's average rank across all boards (DBBP has 2)."""
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS[range_idx]
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD
        hand_2d = np.ascontiguousarray(
            np.array([card_2d_lut[c] for c in hole_cards], dtype=np.int8)
        )

        total_slots = len(board_2d)
        ranks = []
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            board_slice = board_2d[start:end]
            dealt = [c for c in board_slice if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D]
            if len(dealt) < 3:
                continue
            # Pad to 5 cards if needed (turn = 4 cards, need 1 random)
            board_5 = np.empty((5, 2), dtype=np.int8)
            for i, c in enumerate(dealt):
                board_5[i] = c
            if len(dealt) < 5:
                # For turn (4 cards), use a neutral padding — just evaluate with dealt cards
                # Actually, the evaluator needs 5 cards. Use the dealt cards and pad with
                # a deterministic extra card that doesn't conflict.
                hand_1d = set(hole_cards.tolist())
                board_1d = set()
                for c in dealt:
                    board_1d.add(int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))
                for pad_c in range(52):
                    if pad_c not in hand_1d and pad_c not in board_1d:
                        board_5[len(dealt)] = card_2d_lut[pad_c]
                        if len(dealt) + 1 < 5:
                            # Need another pad card (flop = 3 cards, need 2 more)
                            board_1d.add(pad_c)
                            for pad_c2 in range(pad_c + 1, 52):
                                if pad_c2 not in hand_1d and pad_c2 not in board_1d:
                                    board_5[len(dealt) + 1] = card_2d_lut[pad_c2]
                                    break
                        break
            board_5 = np.ascontiguousarray(board_5)
            rank = self._hand_eval.get_hand_rank_52_plo(hand_2d, board_5)
            ranks.append(rank)

        if not ranks:
            return 0
        return sum(ranks) / len(ranks)

    def _compute_rank_boundaries(self, flop_board):
        """
        Compute percentile boundaries for rank-based bucketing.
        Evaluates all valid hands on the flop boards to get a rank distribution,
        then returns the percentile cutoff values for n_buckets bins.
        """
        valid_mask = self._card_abs.get_blocked_mask(flop_board)
        valid_indices = np.where(valid_mask)[0]
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD

        # Split into individual boards
        boards_5 = []
        total_slots = len(flop_board)
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            dealt = np.array(
                [c for c in flop_board[start:end] if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D],
                dtype=np.int8
            )
            if len(dealt) > 0:
                # Pad to 5 cards for evaluation
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

        # Evaluate a sample of hands to get rank distribution
        sample_size = min(5000, len(valid_indices))
        sample_idx = np.random.choice(valid_indices, size=sample_size, replace=False)
        ranks = []
        for ridx in sample_idx:
            hand_2d = np.ascontiguousarray(
                np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
            )
            avg_rank = 0.0
            for board in boards_5:
                avg_rank += self._hand_eval.get_hand_rank_52_plo(hand_2d, board)
            avg_rank /= len(boards_5)
            ranks.append(avg_rank)

        ranks = np.array(ranks)
        # Compute internal percentile boundaries (n_buckets - 1 boundaries)
        percentiles = np.linspace(0, 100, self._n_buckets + 1)[1:-1]
        boundaries = np.percentile(ranks, percentiles)
        return boundaries

    def _compute_rank_boundaries_2d(self, flop_board):
        """
        Compute separate percentile boundaries for each board (DBBP).
        Returns (boundaries_b1, boundaries_b2) each with n_buckets_per_board-1 values.
        """
        valid_mask = self._card_abs.get_blocked_mask(flop_board)
        valid_indices = np.where(valid_mask)[0]
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD

        # Split and pad boards
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
                rank = self._hand_eval.get_hand_rank_52_plo(hand_2d, board)
                ranks_per_board[bi].append(rank)

        npb = self._n_buckets_per_board
        percentiles = np.linspace(0, 100, npb + 1)[1:-1]

        boundaries = []
        for bi in range(len(boards_5)):
            r = np.array(ranks_per_board[bi])
            boundaries.append(np.percentile(r, percentiles))

        # Return exactly 2 boundary arrays for DBBP
        if len(boundaries) >= 2:
            return boundaries[0], boundaries[1]
        return boundaries[0], boundaries[0]
