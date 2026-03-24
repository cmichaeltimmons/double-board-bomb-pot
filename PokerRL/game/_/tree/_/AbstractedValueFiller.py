"""
Counterfactual value computation in bucket space for abstracted game trees.
Handles PLO terminal equity via bucket equity matrices.
"""

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.tree._.nodes import PlayerActionNode


class AbstractedValueFiller:

    def __init__(self, tree, env_bldr, card_abstraction, n_buckets):
        self._tree = tree
        self._env_bldr = env_bldr
        self._card_abs = card_abstraction
        self._n_buckets = n_buckets

        self._hand_eval = card_abstraction._hand_eval

        # Pre-build terminal equity data during first CFV computation
        self._terminal_data_built = False

    def compute_cf_values_heads_up(self, node):
        """
        Compute counterfactual values for all nodes in bucket space.
        Must be called on the root node.
        """
        assert self._tree.n_seats == 2

        if not self._terminal_data_built:
            self._build_terminal_data(node)
            self._terminal_data_built = True

        self._compute_cfv(node)

    def _compute_cfv(self, node):
        if node.is_terminal:
            assert isinstance(node, PlayerActionNode)
            self._compute_terminal_ev(node)
        else:
            N_ACTIONS = len(node.children)
            ev_all = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._n_buckets), dtype=np.float32)
            ev_br_all = np.zeros(shape=(N_ACTIONS, self._tree.n_seats, self._n_buckets), dtype=np.float32)

            for i, child in enumerate(node.children):
                self._compute_cfv(child)
                ev_all[i] = child.ev
                ev_br_all[i] = child.ev_br

            if node.p_id_acting_next == self._tree.CHANCE_ID:
                # Sum over chance outcomes (already weighted by transition probs in reach)
                node.ev = np.sum(ev_all, axis=0)
                node.ev_br = np.sum(ev_br_all, axis=0)
            else:
                node.ev = np.zeros(shape=(self._tree.n_seats, self._n_buckets), dtype=np.float32)
                node.ev_br = np.zeros(shape=(self._tree.n_seats, self._n_buckets), dtype=np.float32)

                plyr = node.p_id_acting_next
                opp = 1 - plyr

                # Player's EV = weighted sum of children EVs by strategy
                # strategy.T is (N_ACTIONS, n_buckets), ev_all[:, plyr] is (N_ACTIONS, n_buckets)
                node.ev[plyr] = np.sum(node.strategy.T * ev_all[:, plyr], axis=0)
                # Opponent's EV = sum of children EVs (opponent's reach already accounts for acting player's strategy)
                node.ev[opp] = np.sum(ev_all[:, opp], axis=0)

                # Best response
                node.ev_br[opp] = np.sum(ev_br_all[:, opp], axis=0)
                node.ev_br[plyr] = np.max(ev_br_all[:, plyr], axis=0)

        # Weighted by reach probs for exploitability
        node.ev_weighted = node.ev * node.reach_probs
        node.ev_br_weighted = node.ev_br * node.reach_probs
        node.epsilon = node.ev_br_weighted - node.ev_weighted
        node.exploitability = np.sum(node.epsilon, axis=1)

    def _compute_terminal_ev(self, node):
        """
        Compute EV at terminal nodes in bucket space using pre-computed equity data.
        """
        pot = node.env_state[EnvDictIdxs.main_pot]
        half_pot = pot / 2.0

        node.ev = np.zeros(shape=(self._tree.n_seats, self._n_buckets), dtype=np.float32)

        if node.action == Poker.FOLD:
            # Player who folded loses. The other player wins.
            folder = node.p_id_acted_last
            winner = 1 - folder

            # Winner's EV per bucket = half_pot * sum of opponent's reach in non-overlapping buckets
            # Folder's EV per bucket = -half_pot * sum of opponent's reach in non-overlapping buckets
            # In bucket space, we approximate by assuming buckets don't overlap
            # (hands in different buckets have no card conflicts on average)
            opp_reach_total = np.sum(node.reach_probs[folder])
            node.ev[winner] = half_pot * opp_reach_total
            node.ev[folder] = -half_pot * np.sum(node.reach_probs[winner])

            # Better approximation using the bucket interaction matrix if available
            if hasattr(node, 'bucket_interaction') and node.bucket_interaction is not None:
                for p in range(self._tree.n_seats):
                    opp = 1 - p
                    # interaction[b] = sum of opponent reach for hands compatible with bucket b
                    node.ev[p] = node.bucket_interaction @ node.reach_probs[opp]
                    if p == folder:
                        node.ev[p] *= -half_pot
                    else:
                        node.ev[p] *= half_pot

        else:
            # Showdown: use bucket equity matrix
            if hasattr(node, 'bucket_equity') and node.bucket_equity is not None:
                # bucket_equity: (n_buckets, n_buckets) matrix where
                # entry [b_i, b_j] = expected equity of bucket b_i vs bucket b_j
                # Range: [-1, 1] where 1 = always win, -1 = always lose
                for p in range(self._tree.n_seats):
                    opp = 1 - p
                    # EV[p, bucket] = half_pot * sum_over_opp_buckets(equity[bucket, opp_bucket] * opp_reach[opp_bucket])
                    node.ev[p] = half_pot * (node.bucket_equity @ node.reach_probs[opp])
            else:
                # Fallback: zero EV (should not happen if terminal data is built)
                pass

        node.ev_br = np.copy(node.ev)

    # -------------------------------------------------------------------------
    # Pre-build terminal data
    # -------------------------------------------------------------------------

    def _build_terminal_data(self, node):
        """Recursively build equity matrices for terminal nodes."""
        if node.is_terminal:
            if node.action == Poker.FOLD:
                self._build_fold_interaction(node)
            else:
                self._build_showdown_equity(node)
            return

        for child in node.children:
            self._build_terminal_data(child)

    def _build_fold_interaction(self, node):
        """
        Build bucket interaction matrix for fold terminals.
        interaction[b_i, b_j] = fraction of hands in bucket b_j that
        don't share cards with hands in bucket b_i.
        """
        board_2d = node.env_state[EnvDictIdxs.board_2d]
        buckets = self._get_buckets_for_board(board_2d)
        hole_cards = self._card_abs._lut.LUT_IDX_2_HOLE_CARDS

        valid_mask = self._card_abs.get_blocked_mask(board_2d)

        # Build interaction matrix: (n_buckets, n_buckets)
        # interaction[bi, bj] = avg compatibility between hands in bi and bj
        interaction = np.ones((self._n_buckets, self._n_buckets), dtype=np.float32)

        # For efficiency, sample representative hands per bucket
        bucket_hands = [[] for _ in range(self._n_buckets)]
        for ridx in range(self._rules_range_size):
            if valid_mask[ridx] and buckets[ridx] >= 0:
                bucket_hands[buckets[ridx]].append(ridx)

        # For each bucket pair, estimate compatibility via sampling
        MAX_SAMPLES = 50
        for bi in range(self._n_buckets):
            if len(bucket_hands[bi]) == 0:
                interaction[bi, :] = 0.0
                continue
            sample_i = bucket_hands[bi][:MAX_SAMPLES]
            cards_i_sets = [set(hole_cards[h].tolist()) for h in sample_i]

            for bj in range(self._n_buckets):
                if len(bucket_hands[bj]) == 0:
                    interaction[bi, bj] = 0.0
                    continue
                sample_j = bucket_hands[bj][:MAX_SAMPLES]

                compatible = 0
                total = 0
                for ci in cards_i_sets:
                    for hj in sample_j:
                        cj = set(hole_cards[hj].tolist())
                        if not (ci & cj):
                            compatible += 1
                        total += 1

                interaction[bi, bj] = compatible / total if total > 0 else 0.0

        node.bucket_interaction = interaction

    def _build_showdown_equity(self, node):
        """
        Build bucket equity matrix for showdown terminals.
        equity[b_i, b_j] = expected equity of hands in bucket b_i vs hands in bucket b_j.
        Range: [-1, 1].
        Handles dual-board (DBBP) by averaging equity across both boards (50/50 pot split).
        """
        board_2d = node.env_state[EnvDictIdxs.board_2d]
        buckets = self._get_buckets_for_board(board_2d)
        hole_cards = self._card_abs._lut.LUT_IDX_2_HOLE_CARDS
        card_2d_lut = self._card_abs._lut.LUT_1DCARD_2_2DCARD
        valid_mask = self._card_abs.get_blocked_mask(board_2d)

        # Split into individual 5-card boards (handles single-board and dual-board)
        boards = self._card_abs._split_boards(board_2d)

        # Evaluate hand rank on each board for all valid hands
        # hand_ranks_per_board[board_idx][range_idx] = rank
        hand_ranks_per_board = []
        for board in boards:
            ranks = np.full(self._rules_range_size, -1, dtype=np.int32)
            if len(board) == 5:
                for ridx in range(self._rules_range_size):
                    if valid_mask[ridx]:
                        hand_2d = np.array([card_2d_lut[c] for c in hole_cards[ridx]], dtype=np.int8)
                        ranks[ridx] = self._hand_eval.get_hand_rank_52_plo(hand_2d, board)
            hand_ranks_per_board.append(ranks)

        # Group hands by bucket
        bucket_hands = [[] for _ in range(self._n_buckets)]
        for ridx in range(self._rules_range_size):
            if valid_mask[ridx] and buckets[ridx] >= 0:
                bucket_hands[buckets[ridx]].append(ridx)

        # Build equity matrix using sampling for large buckets
        equity = np.zeros((self._n_buckets, self._n_buckets), dtype=np.float32)
        MAX_SAMPLES = 100
        n_boards = len(boards)

        for bi in range(self._n_buckets):
            if len(bucket_hands[bi]) == 0:
                continue
            sample_i = bucket_hands[bi]
            if len(sample_i) > MAX_SAMPLES:
                idxs = np.random.choice(len(sample_i), MAX_SAMPLES, replace=False)
                sample_i = [sample_i[idx] for idx in idxs]

            for bj in range(self._n_buckets):
                if len(bucket_hands[bj]) == 0:
                    continue
                sample_j = bucket_hands[bj]
                if len(sample_j) > MAX_SAMPLES:
                    idxs = np.random.choice(len(sample_j), MAX_SAMPLES, replace=False)
                    sample_j = [sample_j[idx] for idx in idxs]

                wins = 0.0
                total = 0
                for ridx_i in sample_i:
                    cards_i = set(hole_cards[ridx_i].tolist())
                    for ridx_j in sample_j:
                        cards_j = set(hole_cards[ridx_j].tolist())
                        if cards_i & cards_j:
                            continue
                        # Average equity across all boards (DBBP: 50/50 split)
                        matchup_equity = 0.0
                        for board_ranks in hand_ranks_per_board:
                            ri = board_ranks[ridx_i]
                            rj = board_ranks[ridx_j]
                            if ri > rj:
                                matchup_equity += 1.0
                            elif ri == rj:
                                matchup_equity += 0.5
                        matchup_equity /= n_boards
                        wins += matchup_equity
                        total += 1

                if total > 0:
                    equity[bi, bj] = 2.0 * (wins / total) - 1.0

        node.bucket_equity = equity

    @property
    def _rules_range_size(self):
        return self._env_bldr.rules.RANGE_SIZE

    def _get_buckets_for_board(self, board_2d):
        """Get bucket assignments for a board state."""
        n_dealt = sum(1 for c in board_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)
        if n_dealt == 0:
            return self._card_abs.get_preflop_buckets()
        else:
            return self._card_abs.get_postflop_buckets(board_2d)
