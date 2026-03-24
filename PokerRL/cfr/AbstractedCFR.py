"""
Tabular CFR with card abstraction for PLO.
Standalone class — no neural networks, no ChiefBase dependency.
Supports Vanilla CFR, CFR+, and Linear CFR variants.
"""

import json
import os
import time

import numpy as np

from PokerRL.game.card_abstraction import CardAbstraction
from PokerRL.game._.tree.AbstractedPublicTree import AbstractedPublicTree
from PokerRL.game.wrappers import HistoryEnvBuilder
from PokerRL.rl.rl_util import get_env_cls_from_str


class AbstractedCFR:

    def __init__(self,
                 name,
                 game_cls,
                 agent_bet_set,
                 n_buckets=200,
                 variant='plus',
                 starting_stack_sizes=None,
                 n_rollouts=5000,
                 cache_dir='./abstraction_cache'):
        """
        Args:
            name:                   Name for logging/saving
            game_cls:               Game class (PLO, DoubleBoardBombPotPLO, etc.)
            agent_bet_set:          Bet set from bet_sets.py (e.g. POT_ONLY)
            n_buckets:              Number of equity buckets
            variant:                'vanilla', 'plus', or 'linear'
            starting_stack_sizes:   List of stack sizes per seat (default: game default)
            n_rollouts:             Monte Carlo rollouts for equity estimation
            cache_dir:              Directory for cached abstraction data
        """
        self._name = name
        self._n_buckets = n_buckets
        self._variant = variant
        self._n_seats = 2

        if starting_stack_sizes is None:
            starting_stack_sizes = [game_cls.DEFAULT_STACK_SIZE] * self._n_seats

        self._game_cls_str = game_cls.__name__

        # Build environment — limit games don't take bet_sizes_list_as_frac_of_pot
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

        # Build card abstraction
        self._card_abs = CardAbstraction(
            rules=self._env_bldr.rules,
            lut_holder=self._env_bldr.lut_holder,
            n_buckets=n_buckets,
            n_rollouts=n_rollouts,
            cache_dir=cache_dir,
        )

        # Build abstracted game tree
        self._tree = AbstractedPublicTree(
            env_bldr=self._env_bldr,
            stack_size=starting_stack_sizes,
            stop_at_street=None,
            card_abstraction=self._card_abs,
            n_buckets=n_buckets,
        )

        self._iter_counter = 0
        self._metrics = []

    @property
    def name(self):
        return self._name

    def reset(self):
        """Build tree and initialize uniform strategies."""
        print("Building game tree...")
        t0 = time.time()
        self._tree.build_tree()
        dt = time.time() - t0
        print("Tree built: {} nodes ({} non-terminal) in {:.1f}s".format(
            self._tree.n_nodes, self._tree.n_nonterm, dt))

        self._iter_counter = 0

        # Initialize data dict at each player action node
        for p in range(self._n_seats):
            self._reset_player(p)

        # Fill uniform strategies and compute initial values
        self._tree.fill_uniform_random()
        self._tree.compute_ev()

    def iteration(self):
        """Run one CFR iteration (updates both players)."""
        for p in range(self._n_seats):
            self._tree.compute_ev()
            self._compute_regrets(p)
            self._compute_new_strategy(p)
            self._tree.update_reach_probs()
            self._add_strategy_to_average(p)

        self._iter_counter += 1

    def get_exploitability(self):
        """Compute and return current strategy exploitability in bucket space."""
        self._tree.compute_ev()
        if self._tree.root.exploitability is not None:
            return float(np.sum(self._tree.root.exploitability)) / self._n_seats
        return float('inf')

    def get_avg_exploitability(self):
        """Compute exploitability of the average strategy."""
        # Fill tree with average strategy, compute EV, measure exploitability
        self._fill_avg_strategy()
        self._tree.update_reach_probs()
        self._tree.compute_ev()
        if self._tree.root.exploitability is not None:
            expl = float(np.sum(self._tree.root.exploitability)) / self._n_seats
        else:
            expl = float('inf')

        # Restore current strategy
        self._restore_current_strategy()
        self._tree.update_reach_probs()
        return expl

    def save_strategy(self, path):
        """Save average strategy tables to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        strategies = {}
        node_id = [0]

        def _collect(node):
            if node.is_terminal:
                return
            if node.p_id_acting_next != self._tree.CHANCE_ID and node.data is not None:
                avg = node.data.get('avg_strat')
                if avg is not None:
                    strategies[str(node_id[0])] = avg
            node_id[0] += 1
            for c in node.children:
                _collect(c)

        _collect(self._tree.root)
        np.savez_compressed(path, **strategies)
        print("Saved strategy to {}".format(path))

    def save_metrics(self, path):
        """Save training metrics to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._metrics, f, indent=2)

    def log_iteration(self, iteration, elapsed):
        """Log metrics for this iteration."""
        expl = self.get_exploitability()
        entry = {
            'iteration': iteration,
            'exploitability': expl,
            'time_seconds': elapsed,
            'variant': self._variant,
        }
        self._metrics.append(entry)
        return expl

    # -------------------------------------------------------------------------
    # CFR Core
    # -------------------------------------------------------------------------

    def _compute_regrets(self, p_id):
        def _fill(node):
            if node.p_id_acting_next == p_id and not node.is_terminal:
                N = len(node.children)

                # EV of each action
                ev_all_actions = np.zeros(shape=(self._n_buckets, N), dtype=np.float32)
                for i, child in enumerate(node.children):
                    ev_all_actions[:, i] = child.ev[p_id]

                # EV under current strategy
                strat_ev = node.ev[p_id]
                strat_ev_expanded = np.expand_dims(strat_ev, axis=-1).repeat(N, axis=-1)

                # Compute regrets based on variant
                if self._iter_counter == 0:
                    node.data['regret'] = ev_all_actions - strat_ev_expanded
                else:
                    if self._variant == 'vanilla':
                        node.data['regret'] = ev_all_actions - strat_ev_expanded + node.data['regret']
                    elif self._variant == 'plus':
                        # CFR+: clamp regrets to non-negative
                        node.data['regret'] = np.maximum(
                            ev_all_actions - strat_ev_expanded + node.data['regret'], 0
                        )
                    elif self._variant == 'linear':
                        # Linear CFR: weight by iteration
                        t = self._iter_counter
                        node.data['regret'] = (
                            (ev_all_actions - strat_ev_expanded) +
                            node.data['regret'] * (t / (t + 1))
                        )

            for c in node.children:
                _fill(c)

        _fill(self._tree.root)

    def _compute_new_strategy(self, p_id):
        def _fill(node):
            if node.p_id_acting_next == p_id and not node.is_terminal:
                N = len(node.children)
                regret = node.data['regret']

                capped_reg = np.maximum(regret, 0)
                reg_sum = np.expand_dims(np.sum(capped_reg, axis=1), axis=1).repeat(N, axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    node.strategy = np.where(
                        reg_sum > 0.0,
                        capped_reg / reg_sum,
                        np.full(shape=(self._n_buckets, N), fill_value=1.0 / N, dtype=np.float32)
                    )

            for c in node.children:
                _fill(c)

        _fill(self._tree.root)

    def _add_strategy_to_average(self, p_id):
        def _fill(node):
            if node.p_id_acting_next == p_id and not node.is_terminal:
                contrib = node.strategy * np.expand_dims(node.reach_probs[p_id], axis=1)

                if self._variant == 'linear':
                    # Linear CFR: weight contribution by iteration
                    contrib *= (self._iter_counter + 1)

                if self._iter_counter > 0 and node.data.get('avg_strat_sum') is not None:
                    node.data['avg_strat_sum'] += contrib
                else:
                    node.data['avg_strat_sum'] = np.copy(contrib)

                s = np.expand_dims(np.sum(node.data['avg_strat_sum'], axis=1), axis=1)
                N = len(node.children)

                with np.errstate(divide='ignore', invalid='ignore'):
                    node.data['avg_strat'] = np.where(
                        s == 0,
                        np.full(shape=N, fill_value=1.0 / N),
                        node.data['avg_strat_sum'] / s
                    )

            for c in node.children:
                _fill(c)

        _fill(self._tree.root)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _reset_player(self, p_id):
        def _reset(node):
            if node.p_id_acting_next == p_id and not node.is_terminal:
                node.data = {
                    'regret': None,
                    'avg_strat': None,
                    'avg_strat_sum': None,
                    '_current_strat': None,
                }
                node.strategy = None
            for c in node.children:
                _reset(c)

        _reset(self._tree.root)

    def _fill_avg_strategy(self):
        """Replace current strategies with average strategies (save current first)."""
        def _fill(node):
            if not node.is_terminal and node.p_id_acting_next != self._tree.CHANCE_ID:
                if node.data is not None and node.data.get('avg_strat') is not None:
                    node.data['_current_strat'] = np.copy(node.strategy)
                    node.strategy = np.copy(node.data['avg_strat'])
            for c in node.children:
                _fill(c)

        _fill(self._tree.root)

    def _restore_current_strategy(self):
        """Restore current strategies from saved copies."""
        def _restore(node):
            if not node.is_terminal and node.p_id_acting_next != self._tree.CHANCE_ID:
                if node.data is not None and node.data.get('_current_strat') is not None:
                    node.strategy = node.data['_current_strat']
                    node.data['_current_strat'] = None
            for c in node.children:
                _restore(c)

        _restore(self._tree.root)
