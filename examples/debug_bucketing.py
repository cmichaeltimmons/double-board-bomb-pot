"""Debug script to isolate where bucketing is slow."""
import time
import sys
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from PokerRL.game.games import DoubleBoardBombPotPLO
from PokerRL.game import bet_sets
from PokerRL.game.wrappers import HistoryEnvBuilder
from PokerRL.rl.rl_util import get_env_cls_from_str
from PokerRL.game.card_abstraction import CardAbstraction
from PokerRL.game.Poker import Poker

print("1. Building env...")
t0 = time.time()
env_args = DoubleBoardBombPotPLO.ARGS_CLS(
    n_seats=2,
    starting_stack_sizes_list=[10000, 10000],
    bet_sizes_list_as_frac_of_pot=bet_sets.PL_2,
)
env_bldr = HistoryEnvBuilder(
    env_cls=get_env_cls_from_str('DoubleBoardBombPotPLO'),
    env_args=env_args,
)
print("   Done: {:.1f}s".format(time.time() - t0))

print("2. Creating card abstraction...")
t0 = time.time()
ca = CardAbstraction(
    rules=env_bldr.rules,
    lut_holder=env_bldr.lut_holder,
    n_buckets=10,
    n_rollouts=10,
    cache_dir='/tmp/debug_cache',
)
print("   Done: {:.1f}s".format(time.time() - t0))

print("3. Creating env and resetting...")
t0 = time.time()
env = env_bldr.get_new_env(is_evaluating=True)
env.reset()
print("   Board shape: {}".format(env.board.shape))
print("   Board: {}".format(env.board))
print("   Round: {}".format(env.current_round))
print("   Done: {:.1f}s".format(time.time() - t0))

print("4. Testing hand evaluator...")
t0 = time.time()
try:
    from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval
    he = CppHandeval()
    print("   Using C++ evaluator")
except Exception as e:
    print("   C++ failed: {}".format(e))
    from PokerRL.game._.cpp_wrappers.PythonHandeval import PythonHandeval
    he = PythonHandeval()
    print("   Using Python fallback")

lut = env_bldr.lut_holder
hole_cards = lut.LUT_IDX_2_HOLE_CARDS
card_2d = lut.LUT_1DCARD_2_2DCARD
hand_2d = np.array([card_2d[c] for c in hole_cards[0]], dtype=np.int8)

# Build a full 5-card board for testing (C++ evaluator requires exactly 5 cards)
board_2d = np.copy(env.board)
dealt = np.array([c for c in board_2d[:5] if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D], dtype=np.int8)
print("   Dealt board cards (board 1): {} cards".format(len(dealt)))

# Pad to 5 cards with random remaining cards for testing
all_board_1d = set()
for c in board_2d:
    if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
        all_board_1d.add(int(lut.LUT_2DCARD_2_1DCARD[c[0], c[1]]))
my_cards_1d = set(hole_cards[0].tolist())
remaining = [c for c in range(52) if c not in my_cards_1d and c not in all_board_1d]
n_pad = 5 - len(dealt)
if n_pad > 0:
    pad_1d = np.random.choice(remaining, size=n_pad, replace=False)
    test_board_5 = np.concatenate([dealt, np.array([card_2d[c] for c in pad_1d], dtype=np.int8)])
    print("   Padded to 5 cards for eval test")
else:
    test_board_5 = dealt

print("   Test board: {}".format(test_board_5))
rank = he.get_hand_rank_52_plo(hand_2d, test_board_5)
print("   Test rank: {}".format(rank))
print("   Done: {:.4f}s".format(time.time() - t0))

print("5. Timing 1000 PLO evaluations...")
t0 = time.time()
for i in range(1000):
    he.get_hand_rank_52_plo(hand_2d, test_board_5)
dt = time.time() - t0
print("   1000 evals: {:.3f}s ({:.1f} PLO evals/sec)".format(dt, 1000 / dt))

print("6. Testing MC equity for 100 hands (10 rollouts each)...")
t0 = time.time()
deck_cards = list(range(52))

for ridx in range(100):
    h_cards_1d = set(hole_cards[ridx].tolist())
    h_hand_2d = np.array([card_2d[c] for c in hole_cards[ridx]], dtype=np.int8)
    rem = [c for c in deck_cards if c not in h_cards_1d and c not in all_board_1d]

    for _ in range(10):
        drawn = np.random.choice(rem, size=4 + n_pad, replace=False)
        opp_1d = drawn[:4]
        extra = drawn[4:]
        full_board = np.concatenate([dealt, np.array([card_2d[c] for c in extra], dtype=np.int8)])
        opp_2d = np.array([card_2d[c] for c in opp_1d], dtype=np.int8)
        he.get_hand_rank_52_plo(h_hand_2d, full_board)
        he.get_hand_rank_52_plo(opp_2d, full_board)

dt = time.time() - t0
print("   100 hands x 10 rollouts x 2 evals = 2000 evals: {:.3f}s".format(dt))

# Extrapolate for different rollout counts
total_hands = 163185
rate = 2000 / dt
for rollouts in [10, 50, 100, 500]:
    total_evals = total_hands * rollouts * 2  # 2 boards
    est_time = total_evals / rate
    print("   Est. bucketing time ({} rollouts, 2 boards): {:.0f}s ({:.1f} min)".format(
        rollouts, est_time, est_time / 60))

print("\nDone!")
