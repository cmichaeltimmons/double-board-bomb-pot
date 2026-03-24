"""
Strategy filler for abstracted (bucketed) game trees.
Mirrors StrategyFiller but uses (n_buckets, ...) arrays instead of (RANGE_SIZE, ...).
"""

import numpy as np

from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
from PokerRL.game._.tree._.nodes import PlayerActionNode, ChanceNode


class AbstractedStrategyFiller:

    def __init__(self, tree, env_bldr, card_abstraction, n_buckets):
        self._tree = tree
        self._env_bldr = env_bldr
        self._card_abs = card_abstraction
        self._n_buckets = n_buckets
        self._chance_filled = False

    def fill_uniform_random(self):
        if not self._chance_filled:
            self._fill_chance_node_strategy(node=self._tree.root)
            self._chance_filled = True

        self._fill_uniform_random(node=self._tree.root)
        self.update_reach_probs()

    def update_reach_probs(self):
        self._update_reach_probs(node=self._tree.root)

    def _fill_uniform_random(self, node):
        if node.is_terminal:
            return

        # Player action nodes and chance nodes with player acting next
        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            n_actions = len(node.children)
            node.strategy = np.full(shape=(self._n_buckets, n_actions),
                                    fill_value=1.0 / float(n_actions),
                                    dtype=np.float32)

        for c in node.children:
            self._fill_uniform_random(node=c)

    def _update_reach_probs(self, node):
        if node.is_terminal:
            return

        if isinstance(node, ChanceNode) or (isinstance(node, PlayerActionNode)
                                            and (not node.is_terminal)
                                            and node.p_id_acting_next != self._tree.CHANCE_ID):
            # Player action node: update reach probs through strategy
            for c in node.children:
                c.reach_probs = np.copy(node.reach_probs)
                a_idx = node.allowed_actions.index(c.action)
                c.reach_probs[node.p_id_acting_next] = (
                    node.strategy[:, a_idx] * node.reach_probs[node.p_id_acting_next]
                )

        elif node.p_id_acting_next == self._tree.CHANCE_ID:
            # Chance node: use transition matrices to propagate reach
            for c_idx in range(len(node.children)):
                child = node.children[c_idx]
                if hasattr(node, 'chance_transition') and node.chance_transition is not None:
                    # chance_transition[c_idx] is shape (n_buckets,) giving the
                    # fraction of hands in each bucket that are compatible with this child board
                    transition = node.chance_transition[c_idx]  # (n_buckets,)
                    child.reach_probs = np.copy(node.reach_probs)
                    # Both players' reach probs get scaled by the chance probability
                    for p in range(self._tree.n_seats):
                        child.reach_probs[p] = node.reach_probs[p] * transition
                else:
                    # Fallback: uniform over children
                    child.reach_probs = np.copy(node.reach_probs) / len(node.children)
        else:
            raise TypeError("Unexpected node type: {}".format(type(node)))

        for c in node.children:
            self._update_reach_probs(node=c)

    def _fill_chance_node_strategy(self, node):
        """
        For chance nodes, compute transition matrices that map parent buckets
        to child boards. Each child gets a transition vector of shape (n_buckets,)
        representing the probability mass that transitions to that child.
        """
        if node.is_terminal:
            return

        if node.p_id_acting_next == self._tree.CHANCE_ID:
            n_children = len(node.children)

            # Compute bucket mapping for each child board
            # Each child corresponds to a different board card being dealt
            parent_board_2d = node.env_state[EnvDictIdxs.board_2d]

            # Get parent buckets (the bucket mapping for the current board)
            parent_buckets = self._get_buckets_for_board(parent_board_2d)
            parent_valid = (parent_buckets >= 0)

            # For each child, compute what fraction of each parent bucket
            # transitions to this child (i.e., is compatible with the new card)
            node.chance_transition = []

            # Count hands per parent bucket for normalization
            parent_bucket_counts = np.zeros(self._n_buckets, dtype=np.float32)
            for b in range(self._n_buckets):
                parent_bucket_counts[b] = np.sum(parent_buckets == b)

            for c_idx in range(n_children):
                child = node.children[c_idx]
                child_board_2d = child.env_state[EnvDictIdxs.board_2d]

                # Which hands are valid for this child board?
                child_valid = self._card_abs.get_blocked_mask(child_board_2d)

                # Transition: for each parent bucket, what fraction of hands
                # in that bucket are still valid in this child?
                transition = np.zeros(self._n_buckets, dtype=np.float32)
                for b in range(self._n_buckets):
                    bucket_mask = (parent_buckets == b)
                    n_in_bucket = np.sum(bucket_mask)
                    if n_in_bucket > 0:
                        n_still_valid = np.sum(bucket_mask & child_valid)
                        transition[b] = n_still_valid / n_in_bucket

                # Normalize so that transitions across all children sum to ~1
                # for each bucket (this accounts for uniform chance over cards)
                # We'll normalize after collecting all children
                node.chance_transition.append(transition)

            # Normalize: for each bucket, the sum of transitions across all children
            # should represent the total probability mass (which gets redistributed)
            total_transition = np.zeros(self._n_buckets, dtype=np.float32)
            for t in node.chance_transition:
                total_transition += t

            for c_idx in range(n_children):
                with np.errstate(divide='ignore', invalid='ignore'):
                    node.chance_transition[c_idx] = np.where(
                        total_transition > 0,
                        node.chance_transition[c_idx] / total_transition,
                        1.0 / n_children
                    )

            # Also set a dummy strategy (not used in reach prop, but needed for structure)
            node.strategy = np.full(shape=(self._n_buckets, n_children),
                                    fill_value=1.0 / float(n_children),
                                    dtype=np.float32)

        for c in node.children:
            self._fill_chance_node_strategy(node=c)

    def _get_buckets_for_board(self, board_2d):
        """Get bucket assignments for a board state. Preflop if no cards dealt."""
        n_dealt = sum(1 for c in board_2d if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D)
        if n_dealt == 0:
            return self._card_abs.get_preflop_buckets()
        else:
            return self._card_abs.get_postflop_buckets(board_2d)


from PokerRL.game.Poker import Poker
