"""
Tabular CFR with card abstraction for PLO / DoubleBoardBombPot.
No neural networks — uses equity bucketing + lookup tables.

Usage:
    python examples/run_tabular_cfr.py --game dbbp --n-buckets 200 --n-iterations 1000 --variant plus
    python examples/run_tabular_cfr.py --game plo --n-buckets 100 --n-iterations 500 --variant vanilla
"""

import argparse
import os
import time

os.environ["OMP_NUM_THREADS"] = "1"

from PokerRL.game.games import PLO, DoubleBoardBombPotPLO
from PokerRL.game import bet_sets
from PokerRL.cfr.AbstractedCFR import AbstractedCFR


def main():
    parser = argparse.ArgumentParser(description='Tabular CFR with card abstraction')
    parser.add_argument('--game', choices=['plo', 'dbbp'], default='dbbp',
                        help='Game type: plo (PLO) or dbbp (Double Board Bomb Pot PLO)')
    parser.add_argument('--n-buckets', type=int, default=200,
                        help='Number of equity buckets (default: 200)')
    parser.add_argument('--n-iterations', type=int, default=1000,
                        help='Number of CFR iterations (default: 1000)')
    parser.add_argument('--variant', choices=['vanilla', 'plus', 'linear'], default='plus',
                        help='CFR variant (default: plus)')
    parser.add_argument('--stack-size', type=int, default=10000,
                        help='Starting stack size (default: 10000)')
    parser.add_argument('--cache-dir', type=str, default='./abstraction_cache',
                        help='Directory for cached abstraction data')
    parser.add_argument('--rollouts', type=int, default=5000,
                        help='Monte Carlo rollouts for equity estimation (default: 5000)')
    parser.add_argument('--log-freq', type=int, default=10,
                        help='Log exploitability every N iterations (default: 10)')
    parser.add_argument('--save-dir', type=str, default='./strategies',
                        help='Directory to save final strategy')
    args = parser.parse_args()

    game_cls = DoubleBoardBombPotPLO if args.game == 'dbbp' else PLO
    game_name = 'DBBP_PLO' if args.game == 'dbbp' else 'PLO'

    print("=" * 60)
    print("Tabular CFR with Card Abstraction")
    print("=" * 60)
    print("Game:       {}".format(game_name))
    print("Buckets:    {}".format(args.n_buckets))
    print("Variant:    CFR{}".format('+' if args.variant == 'plus' else
                                     ' (vanilla)' if args.variant == 'vanilla' else
                                     ' (linear)'))
    print("Iterations: {}".format(args.n_iterations))
    print("Stack:      {}".format(args.stack_size))
    print("Rollouts:   {}".format(args.rollouts))
    print("=" * 60)

    cfr = AbstractedCFR(
        name="TabularCFR_{}_{}b".format(game_name, args.n_buckets),
        game_cls=game_cls,
        agent_bet_set=bet_sets.PL_2,
        n_buckets=args.n_buckets,
        variant=args.variant,
        starting_stack_sizes=[args.stack_size, args.stack_size],
        n_rollouts=args.rollouts,
        cache_dir=args.cache_dir,
    )

    print("\nInitializing...")
    t_start = time.time()
    cfr.reset()
    t_init = time.time() - t_start
    print("Initialization took {:.1f}s\n".format(t_init))

    print("Starting CFR iterations...")
    print("-" * 60)

    for i in range(args.n_iterations):
        t0 = time.time()
        cfr.iteration()
        dt = time.time() - t0

        if (i + 1) % args.log_freq == 0 or i == 0:
            expl = cfr.log_iteration(i + 1, dt)
            print("Iter {:5d} | Exploitability: {:10.6f} | Time: {:.2f}s".format(i + 1, expl, dt))

    print("-" * 60)
    total_time = time.time() - t_start
    print("Training complete. Total time: {:.1f}s".format(total_time))

    # Save
    strategy_path = os.path.join(
        args.save_dir,
        '{}_{}b_{}iter_{}.npz'.format(game_name, args.n_buckets, args.n_iterations, args.variant)
    )
    cfr.save_strategy(strategy_path)

    metrics_path = os.path.join(
        args.save_dir,
        '{}_{}b_{}iter_{}_metrics.json'.format(game_name, args.n_buckets, args.n_iterations, args.variant)
    )
    cfr.save_metrics(metrics_path)

    # Final exploitability of average strategy
    avg_expl = cfr.get_avg_exploitability()
    print("\nFinal average strategy exploitability: {:.6f}".format(avg_expl))


if __name__ == '__main__':
    main()
