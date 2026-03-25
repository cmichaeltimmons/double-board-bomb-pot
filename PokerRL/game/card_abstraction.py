"""
Card abstraction for PLO / DBBP: maps 270K hands to N equity buckets.
Pre-computes equity-based bucketing for preflop and postflop streets.
"""

import os
import hashlib
import numpy as np
from collections import defaultdict

from PokerRL.game.Poker import Poker


def _eval_chunk_exhaustive(args):
    """
    Worker function for parallel exhaustive equity computation.
    Must be at module level for multiprocessing to pickle it.
    """
    (chunk_indices, partial_board, board_completions, n_dealt,
     hole_cards, card_2d_lut, chunk_id, n_chunks) = args

    import sys
    try:
        from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
        hand_eval = CppHandeval()
    except Exception:
        from PokerRL.game._.cpp_wrappers.PythonHandeval import PythonHandeval
        hand_eval = PythonHandeval()

    equities = []
    full_board = np.empty((5, 2), dtype=np.int8)
    full_board[:n_dealt] = partial_board

    for i, ridx in enumerate(chunk_indices):
        if chunk_id == 0 and i % 100 == 0:
            print("    Progress: hand {}/{} (core 1)...".format(i, len(chunk_indices)))
            sys.stdout.flush()

        my_cards_1d = set(hole_cards[ridx].tolist())
        my_hand_2d = np.ascontiguousarray(
            np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
        )

        rank_sum = 0
        count = 0
        for completion in board_completions:
            if any(c in my_cards_1d for c in completion):
                continue
            for j, c in enumerate(completion):
                full_board[n_dealt + j] = card_2d_lut[c]
            full_board_c = np.ascontiguousarray(full_board)
            rank_sum += hand_eval.get_hand_rank_52_plo(my_hand_2d, full_board_c)
            count += 1

        equities.append(rank_sum / count if count > 0 else 0)

    return chunk_indices, equities


class CardAbstraction:
    """
    Maps range_idx -> bucket_id based on hand equity.

    Preflop: groups by sorted rank pattern, estimates equity via Monte Carlo,
             then assigns percentile-based buckets.
    Postflop: computes hand strength against random opponent for a given board,
              then assigns percentile-based buckets. On river, equity is exact.

    For dual-board games (DBBP), uses 2D bucketing: each board gets its own
    bucket, and the final bucket = board1_bucket * n_per_board + board2_bucket.
    This preserves per-board strength information that averaging destroys.
    """

    def __init__(self, rules, lut_holder, n_buckets, n_rollouts=5000,
                 cache_dir='./abstraction_cache', n_buckets_per_board=None):
        self._rules = rules
        self._lut = lut_holder
        self._n_buckets = n_buckets
        self._n_rollouts = n_rollouts
        self._cache_dir = cache_dir

        # For DBBP 2D bucketing: n_buckets_per_board^2 = total buckets
        # If not specified, infer from n_buckets (e.g. 225 -> 15 per board)
        if n_buckets_per_board is not None:
            self._n_buckets_per_board = n_buckets_per_board
        else:
            self._n_buckets_per_board = int(np.sqrt(n_buckets))

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

        For dual boards, uses 2D bucketing: each board gets independent buckets,
        then combined as board1_bucket * n_per_board + board2_bucket.
        This preserves per-board strength (e.g. nuts on board 1 + air on board 2
        is distinct from marginal on both).
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

        # Compute equity per board separately
        per_board_equity = []
        for bi, board in enumerate(boards):
            if len(board) == 5:
                eq = self._compute_river_equity_single(valid_indices, board, all_board_cards_1d)
            else:
                eq = self._compute_exhaustive_equity_single(
                    valid_indices, board, all_board_cards_1d, card_2d_lut, hole_cards)
            per_board_equity.append(eq)

        if n_boards == 1:
            # Single board: standard 1D bucketing
            buckets = self._percentile_bucket(per_board_equity[0], valid_mask)
        else:
            # Dual board: 2D bucketing
            npb = self._n_buckets_per_board
            print("  2D bucketing: {} per board, {} total".format(npb, npb * npb))

            b1 = self._percentile_bucket_n(per_board_equity[0], valid_mask, npb)
            b2 = self._percentile_bucket_n(per_board_equity[1], valid_mask, npb)

            # Combine: bucket = b1 * npb + b2
            buckets = np.full(self._rules.RANGE_SIZE, -1, dtype=np.int16)
            for idx in valid_indices:
                if b1[idx] >= 0 and b2[idx] >= 0:
                    buckets[idx] = b1[idx] * npb + b2[idx]

        print("  Postflop bucketing complete.")
        return buckets

    def _compute_river_equity_single(self, valid_indices, board_5, all_board_cards_1d):
        """Exact equity on a single complete 5-card board (vectorized)."""
        import sys
        hole_cards = self._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._lut.LUT_1DCARD_2_2DCARD
        board_5_c = np.ascontiguousarray(board_5, dtype=np.int8)

        # Step 1: compute hand rank for all valid hands
        print("    Computing hand ranks for {} hands...".format(len(valid_indices)))
        sys.stdout.flush()
        hand_ranks = np.full(self._rules.RANGE_SIZE, -1, dtype=np.int32)
        for i, ridx in enumerate(valid_indices):
            if i % 10000 == 0:
                print("    Ranking hand {}/{}...".format(i, len(valid_indices)))
                sys.stdout.flush()
            hand_2d = np.ascontiguousarray(
                np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
            )
            hand_ranks[ridx] = self._hand_eval.get_hand_rank_52_plo(hand_2d, board_5_c)

        # Step 2: vectorized equity via rank comparison
        # Build blocked-cards set per valid hand for fast overlap detection
        print("    Computing pairwise equity (vectorized)...")
        sys.stdout.flush()
        valid_ranks = hand_ranks[valid_indices]  # (N,)
        valid_hole = hole_cards[valid_indices]    # (N, 4) 1D card indices
        N = len(valid_indices)

        # Build card sets as sorted tuples for fast overlap check using numpy
        # For each pair, hands overlap if they share any card
        # Vectorize: expand hole cards to (N, 4) and check pairwise overlap
        # Process in chunks to avoid memory issues
        hand_equity = np.zeros(self._rules.RANGE_SIZE, dtype=np.float32)
        CHUNK = 1000
        for start in range(0, N, CHUNK):
            if start % 10000 == 0:
                print("    Equity chunk {}/{}...".format(start, N))
                sys.stdout.flush()
            end = min(start + CHUNK, N)
            chunk_ranks = valid_ranks[start:end]  # (C,)
            chunk_holes = valid_hole[start:end]    # (C, 4)

            # Check card overlap: for each (chunk_hand, valid_hand) pair,
            # do they share any card? Check all 4x4 card combinations.
            overlap = np.zeros((end - start, N), dtype=bool)
            for ci in range(4):
                for oi in range(4):
                    overlap |= (chunk_holes[:, ci:ci+1] == valid_hole[:, oi:oi+1].T)

            # Rank comparison: (C, 1) vs (1, N)
            wins = (chunk_ranks[:, None] > valid_ranks[None, :]).astype(np.float32)
            ties = (chunk_ranks[:, None] == valid_ranks[None, :]).astype(np.float32) * 0.5

            # Mask out self and overlapping hands
            mask = ~overlap
            # Also mask self-comparisons
            idx_range = np.arange(start, end)
            for local_i, global_i in enumerate(idx_range):
                mask[local_i, global_i] = False

            total = mask.sum(axis=1).astype(np.float32)
            win_sum = ((wins + ties) * mask).sum(axis=1)

            equity = np.where(total > 0, win_sum / total, 0.5)
            for local_i in range(end - start):
                hand_equity[valid_indices[start + local_i]] = equity[local_i]

        return hand_equity

    def _compute_exhaustive_equity_single(self, valid_indices, partial_board,
                                           all_board_cards_1d, card_2d_lut, hole_cards):
        """
        Exhaustive equity for an incomplete board (flop or turn).
        Enumerates all possible remaining board cards (~870 combos on flop,
        ~30 on turn) and averages hand rank across all completions.
        Parallelized across all available CPU cores.
        """
        import sys
        import itertools
        from multiprocessing import Pool, cpu_count

        n_dealt = len(partial_board)
        n_remaining = 5 - n_dealt

        deck_cards = [c for c in range(self._rules.N_CARDS_IN_DECK) if c not in all_board_cards_1d]
        board_completions = list(itertools.combinations(deck_cards, n_remaining))
        n_completions = len(board_completions)

        n_workers = cpu_count()
        print("    Exhaustive equity: {} board completions, {} hands, {} workers...".format(
            n_completions, len(valid_indices), n_workers))
        sys.stdout.flush()

        hand_equity = np.zeros(self._rules.RANGE_SIZE, dtype=np.float32)

        # Split valid_indices into chunks for parallel processing
        chunk_size = max(1, len(valid_indices) // n_workers)
        chunks = []
        for start in range(0, len(valid_indices), chunk_size):
            end = min(start + chunk_size, len(valid_indices))
            chunks.append(valid_indices[start:end])

        # Prepare shared data for worker processes
        worker_args = [
            (chunk, partial_board, board_completions, n_dealt,
             hole_cards, card_2d_lut, ci, len(chunks))
            for ci, chunk in enumerate(chunks)
        ]

        with Pool(n_workers) as pool:
            results = pool.map(_eval_chunk_exhaustive, worker_args)

        # Merge results
        for chunk_indices, chunk_equities in results:
            for ridx, eq in zip(chunk_indices, chunk_equities):
                hand_equity[ridx] = eq

        return hand_equity

    # -------------------------------------------------------------------------
    # Bucketing
    # -------------------------------------------------------------------------

    def _percentile_bucket(self, equity, valid_mask):
        """
        Assign hands to n_buckets equally-spaced percentile bins based on equity.
        Invalid hands get bucket -1.
        """
        return self._percentile_bucket_n(equity, valid_mask, self._n_buckets)

    def _percentile_bucket_n(self, equity, valid_mask, n):
        """
        Assign hands to n equally-spaced percentile bins based on equity.
        Invalid hands get bucket -1.
        """
        buckets = np.full(self._rules.RANGE_SIZE, -1, dtype=np.int16)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return buckets

        valid_equities = equity[valid_indices]

        # Compute percentile boundaries
        percentiles = np.linspace(0, 100, n + 1)
        boundaries = np.percentile(valid_equities, percentiles)

        # Assign buckets via digitize
        bucket_ids = np.digitize(valid_equities, boundaries[1:-1])
        bucket_ids = np.clip(bucket_ids, 0, n - 1)

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
