"""
Card abstraction for PLO / DBBP: maps 270K hands to N equity buckets.
Pre-computes equity-based bucketing for preflop and postflop streets.
"""

import os
import hashlib
import numpy as np
from collections import defaultdict

from PokerRL.game.Poker import Poker


class CardAbstraction:
    """
    Maps range_idx -> bucket_id based on hand equity.

    Preflop: groups by sorted rank pattern, estimates equity via Monte Carlo,
             then assigns percentile-based buckets.
    Postflop: computes hand strength against random opponent for a given board,
              then assigns percentile-based buckets. On river, equity is exact.
    """

    def __init__(self, rules, lut_holder, n_buckets, n_rollouts=5000, cache_dir='./abstraction_cache'):
        self._rules = rules
        self._lut = lut_holder
        self._n_buckets = n_buckets
        self._n_rollouts = n_rollouts
        self._cache_dir = cache_dir

        self._hand_eval = self._get_hand_evaluator()

        # Preflop bucket map: shape (RANGE_SIZE,), values in [0, n_buckets)
        self._preflop_buckets = None

        # Cache of postflop bucket maps keyed by board hash
        self._postflop_cache = {}

    def _get_hand_evaluator(self):
        """Get hand evaluator, preferring C++ but falling back to Python."""
        try:
            from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
            return CppHandeval()
        except Exception:
            from PokerRL.game._.cpp_wrappers.PythonHandeval import PythonHandeval
            return PythonHandeval()

    @property
    def n_buckets(self):
        return self._n_buckets

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_preflop_buckets(self):
        """Returns preflop bucket assignments, computing if needed."""
        if self._preflop_buckets is None:
            cached = self._load_cache('preflop')
            if cached is not None:
                self._preflop_buckets = cached
            else:
                self._preflop_buckets = self._compute_preflop_buckets()
                self._save_cache('preflop', self._preflop_buckets)
        return self._preflop_buckets

    def get_postflop_buckets(self, board_2d):
        """
        Returns postflop bucket assignments for a given board.

        Args:
            board_2d: np.ndarray of shape (N, 2) with dealt board cards.

        Returns:
            np.ndarray of shape (RANGE_SIZE,) with bucket IDs.
            Blocked hands get bucket -1.
        """
        key = self._board_hash(board_2d)
        if key in self._postflop_cache:
            return self._postflop_cache[key]

        cached = self._load_cache('postflop_' + key)
        if cached is not None:
            self._postflop_cache[key] = cached
            return cached

        buckets = self._compute_postflop_buckets(board_2d)
        self._postflop_cache[key] = buckets
        self._save_cache('postflop_' + key, buckets)
        return buckets

    def get_blocked_mask(self, board_2d):
        """
        Returns boolean mask of shape (RANGE_SIZE,). True = hand is NOT blocked.
        Uses vectorized numpy operations for PLO 4-card hands.
        """
        board_1d = set()
        for c in board_2d:
            if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                c1d = int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]])
                board_1d.add(c1d)

        if len(board_1d) == 0:
            return np.ones(self._rules.RANGE_SIZE, dtype=bool)

        # LUT_IDX_2_HOLE_CARDS: shape (RANGE_SIZE, 4) for PLO
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS  # (RANGE_SIZE, N_HOLE_CARDS)
        mask = np.ones(self._rules.RANGE_SIZE, dtype=bool)
        for bc in board_1d:
            mask &= ~np.any(hole_cards == bc, axis=1)
        return mask

    # -------------------------------------------------------------------------
    # Preflop bucketing
    # -------------------------------------------------------------------------

    def _compute_preflop_buckets(self):
        """
        1. Group hands by sorted rank pattern (ignore suits).
        2. Estimate equity per pattern via Monte Carlo.
        3. Assign percentile-based buckets.
        """
        print("Computing preflop buckets ({} buckets, {} rollouts)...".format(
            self._n_buckets, self._n_rollouts))
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS  # (RANGE_SIZE, 4) 1D card indices
        card_2d = self._lut.LUT_1DCARD_2_2DCARD  # (52, 2) -> [rank, suit]

        # Group by sorted rank tuple
        pattern_to_indices = defaultdict(list)
        for ridx in range(self._rules.RANGE_SIZE):
            cards_1d = hole_cards[ridx]
            ranks = tuple(sorted(int(card_2d[c][0]) for c in cards_1d))
            pattern_to_indices[ranks].append(ridx)

        patterns = list(pattern_to_indices.keys())
        print("  {} canonical rank patterns".format(len(patterns)))

        # Estimate equity per pattern
        pattern_equity = {}
        deck_cards = list(range(self._rules.N_CARDS_IN_DECK))

        for pi, ranks in enumerate(patterns):
            if pi % 2000 == 0:
                print("  Pattern {}/{}...".format(pi, len(patterns)))
            # Pick a representative hand for this pattern
            rep_idx = pattern_to_indices[ranks][0]
            rep_cards_1d = set(hole_cards[rep_idx].tolist())
            rep_hand_2d = np.array([card_2d[c] for c in hole_cards[rep_idx]], dtype=np.int8)

            wins = 0
            total = 0
            remaining = [c for c in deck_cards if c not in rep_cards_1d]

            for _ in range(self._n_rollouts):
                drawn = np.random.choice(remaining, size=self._rules.N_HOLE_CARDS + 5, replace=False)
                opp_1d = drawn[:self._rules.N_HOLE_CARDS]
                board_1d = drawn[self._rules.N_HOLE_CARDS:]

                opp_2d = np.array([card_2d[c] for c in opp_1d], dtype=np.int8)
                board_2d_arr = np.array([card_2d[c] for c in board_1d], dtype=np.int8)

                rank_us = self._hand_eval.get_hand_rank_52_plo(rep_hand_2d, board_2d_arr)
                rank_opp = self._hand_eval.get_hand_rank_52_plo(opp_2d, board_2d_arr)

                if rank_us > rank_opp:
                    wins += 1
                elif rank_us == rank_opp:
                    wins += 0.5
                total += 1

            pattern_equity[ranks] = wins / total if total > 0 else 0.5

        # Assign hand -> equity
        hand_equity = np.zeros(self._rules.RANGE_SIZE, dtype=np.float32)
        for ranks, indices in pattern_to_indices.items():
            eq = pattern_equity[ranks]
            for idx in indices:
                hand_equity[idx] = eq

        # Percentile bucketing
        buckets = self._percentile_bucket(hand_equity, np.ones(self._rules.RANGE_SIZE, dtype=bool))
        print("  Preflop bucketing complete.")
        return buckets

    # -------------------------------------------------------------------------
    # Postflop bucketing
    # -------------------------------------------------------------------------

    def _split_boards(self, board_2d):
        """
        Split a board_2d array into individual 5-card boards.
        For single-board games: returns [board_2d].
        For DBBP (10-card layout): returns [board1(0:5), board2(5:10)].
        Each sub-board only includes dealt cards.
        """
        total_slots = len(board_2d)
        if total_slots <= 5:
            dealt = np.array([c for c in board_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8)
            return [dealt] if len(dealt) > 0 else []

        # Dual board: indices 0-4 = board1, 5-9 = board2
        boards = []
        for start in range(0, total_slots, 5):
            end = min(start + 5, total_slots)
            dealt = np.array([c for c in board_2d[start:end]
                              if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8)
            if len(dealt) > 0:
                boards.append(dealt)
        return boards

    def _compute_postflop_buckets(self, board_2d):
        """
        Compute equity for all non-blocked hands, then bucket.
        Handles both single-board and dual-board (DBBP) games.
        For dual boards, equity is averaged across both boards.
        """
        boards = self._split_boards(board_2d)
        n_boards = len(boards)
        n_board_cards = sum(len(b) for b in boards)
        is_complete = all(len(b) == 5 for b in boards)
        print("Computing postflop buckets ({} board(s), {} cards, complete={})...".format(
            n_boards, n_board_cards, is_complete))

        valid_mask = self.get_blocked_mask(board_2d)
        valid_indices = np.where(valid_mask)[0]
        print("  {} valid hands (of {})".format(len(valid_indices), self._rules.RANGE_SIZE))

        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD

        # Collect all dealt board cards for blocking during MC
        all_board_cards_1d = set()
        for c in board_2d:
            if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                all_board_cards_1d.add(int(self._lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))

        # Compute equity per board, then average
        hand_equity = np.zeros(self._rules.RANGE_SIZE, dtype=np.float32)
        for bi, board in enumerate(boards):
            if len(board) == 5:
                eq = self._compute_river_equity_single(valid_indices, board, all_board_cards_1d)
            else:
                eq = self._compute_mc_equity_single(valid_indices, board, all_board_cards_1d,
                                                     card_2d_lut, hole_cards)
            hand_equity += eq

        hand_equity /= n_boards

        buckets = self._percentile_bucket(hand_equity, valid_mask)
        print("  Postflop bucketing complete.")
        return buckets

    def _compute_river_equity_single(self, valid_indices, board_5, all_board_cards_1d):
        """Exact equity on a single complete 5-card board."""
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD

        hand_ranks = np.full(self._rules.RANGE_SIZE, -1, dtype=np.int32)
        for ridx in valid_indices:
            hand_2d = np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
            hand_ranks[ridx] = self._hand_eval.get_hand_rank_52_plo(hand_2d, board_5)

        hand_equity = np.zeros(self._rules.RANGE_SIZE, dtype=np.float32)
        for ridx in valid_indices:
            my_cards = set(hole_cards[ridx].tolist())
            my_rank = hand_ranks[ridx]
            wins = 0.0
            total = 0
            for oidx in valid_indices:
                if oidx == ridx:
                    continue
                if my_cards & set(hole_cards[oidx].tolist()):
                    continue
                opp_rank = hand_ranks[oidx]
                if my_rank > opp_rank:
                    wins += 1.0
                elif my_rank == opp_rank:
                    wins += 0.5
                total += 1
            hand_equity[ridx] = wins / total if total > 0 else 0.5

        return hand_equity

    def _compute_mc_equity_single(self, valid_indices, partial_board, all_board_cards_1d,
                                   card_2d_lut, hole_cards):
        """Monte Carlo equity for a single incomplete board (flop or turn)."""
        import sys
        n_remaining = 5 - len(partial_board)
        deck_cards = list(range(self._rules.N_CARDS_IN_DECK))

        hand_equity = np.zeros(self._rules.RANGE_SIZE, dtype=np.float32)

        # Pre-allocate contiguous buffers for C++ evaluator compatibility
        full_board_2d = np.empty((5, 2), dtype=np.int8)
        full_board_2d[:len(partial_board)] = partial_board
        opp_2d = np.empty((self._rules.N_HOLE_CARDS, 2), dtype=np.int8)

        for i, ridx in enumerate(valid_indices):
            if i % 1000 == 0:
                print("    MC equity: hand {}/{}...".format(i, len(valid_indices)))
                sys.stdout.flush()

            my_cards_1d = set(hole_cards[ridx].tolist())
            my_hand_2d = np.ascontiguousarray(
                np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
            )
            remaining = [c for c in deck_cards if c not in my_cards_1d and c not in all_board_cards_1d]

            wins = 0.0
            total = 0
            for _ in range(self._n_rollouts):
                drawn = np.random.choice(remaining,
                                         size=self._rules.N_HOLE_CARDS + n_remaining,
                                         replace=False)
                opp_1d = drawn[:self._rules.N_HOLE_CARDS]
                extra_board_1d = drawn[self._rules.N_HOLE_CARDS:]

                # Fill pre-allocated buffers (avoids np.concatenate stride issues)
                for j, c in enumerate(extra_board_1d):
                    full_board_2d[len(partial_board) + j] = card_2d_lut[c]
                for j, c in enumerate(opp_1d):
                    opp_2d[j] = card_2d_lut[c]

                rank_us = self._hand_eval.get_hand_rank_52_plo(my_hand_2d, full_board_2d)
                rank_opp = self._hand_eval.get_hand_rank_52_plo(opp_2d, full_board_2d)

                if rank_us > rank_opp:
                    wins += 1.0
                elif rank_us == rank_opp:
                    wins += 0.5
                total += 1

            hand_equity[ridx] = wins / total if total > 0 else 0.5

        return hand_equity

    # -------------------------------------------------------------------------
    # Bucketing
    # -------------------------------------------------------------------------

    def _percentile_bucket(self, equity, valid_mask):
        """
        Assign hands to n_buckets equally-spaced percentile bins based on equity.
        Invalid hands get bucket -1.
        """
        buckets = np.full(self._rules.RANGE_SIZE, -1, dtype=np.int16)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return buckets

        valid_equities = equity[valid_indices]

        # Compute percentile boundaries
        percentiles = np.linspace(0, 100, self._n_buckets + 1)
        boundaries = np.percentile(valid_equities, percentiles)

        # Assign buckets via digitize
        # boundaries[1:-1] gives the internal bin edges
        bucket_ids = np.digitize(valid_equities, boundaries[1:-1])
        bucket_ids = np.clip(bucket_ids, 0, self._n_buckets - 1)

        buckets[valid_indices] = bucket_ids.astype(np.int16)
        return buckets

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _board_hash(self, board_2d):
        """Deterministic hash for a board state."""
        cards = []
        for c in board_2d:
            if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                cards.append((int(c[0]), int(c[1])))
        cards.sort()
        return hashlib.md5(repr(cards).encode()).hexdigest()[:12]

    def _cache_path(self, name):
        return os.path.join(
            self._cache_dir,
            '{}_b{}_r{}.npy'.format(name, self._n_buckets, self._n_rollouts)
        )

    def _load_cache(self, name):
        path = self._cache_path(name)
        if os.path.exists(path):
            print("  Loading cached {} from {}".format(name, path))
            return np.load(path)
        return None

    def _save_cache(self, name, data):
        os.makedirs(self._cache_dir, exist_ok=True)
        path = self._cache_path(name)
        np.save(path, data)
        print("  Saved {} to {}".format(name, path))
