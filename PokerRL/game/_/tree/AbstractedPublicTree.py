"""
PublicTree subclass that stores bucket-dimensioned arrays instead of RANGE_SIZE arrays.
Uses CardAbstraction for equity bucketing and abstracted fillers for strategy/value computation.
"""

import numpy as np

from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.game._.tree._.AbstractedStrategyFiller import AbstractedStrategyFiller
from PokerRL.game._.tree._.AbstractedValueFiller import AbstractedValueFiller


class AbstractedPublicTree(PublicTree):

    def __init__(self, env_bldr, stack_size, stop_at_street, card_abstraction, n_buckets):
        self._card_abs = card_abstraction
        self._n_buckets = n_buckets

        # Call parent init — this sets up env, builds nothing yet
        super().__init__(
            env_bldr=env_bldr,
            stack_size=stack_size,
            stop_at_street=stop_at_street,
            put_out_new_round_after_limit=False,
            is_debugging=False,
        )

        # Replace fillers with abstracted versions
        self._value_filler = AbstractedValueFiller(
            tree=self, env_bldr=env_bldr,
            card_abstraction=card_abstraction, n_buckets=n_buckets
        )
        self._strategy_filler = AbstractedStrategyFiller(
            tree=self, env_bldr=env_bldr,
            card_abstraction=card_abstraction, n_buckets=n_buckets
        )

    @property
    def n_buckets(self):
        return self._n_buckets

    @property
    def card_abstraction(self):
        return self._card_abs

    def build_tree(self):
        """Build tree then override root reach_probs to bucket dimensions."""
        # Build the tree structure (reuses parent logic entirely)
        super().build_tree()

        # Override root reach_probs from (n_seats, RANGE_SIZE) to (n_seats, n_buckets)
        self.root.reach_probs = np.full(
            shape=(self._env_bldr.N_SEATS, self._n_buckets),
            fill_value=1.0 / float(self._n_buckets),
            dtype=np.float32
        )
