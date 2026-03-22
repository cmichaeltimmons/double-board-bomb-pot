"""
Watch the Double Board Bomb Pot agent play against itself.
Both seats are controlled by the AI.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.Poker import Poker


RANK_STR = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A'}
SUIT_SYMBOL = {0:'\u2665', 1:'\u2666', 2:'\u2660', 3:'\u2663'}

ACTION_NAMES = ['FOLD', 'CHECK/CALL', 'BET 33% POT', 'BET 50% POT', 'BET POT']


def card_str(card_2d):
    if card_2d[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
        return '--'
    return RANK_STR[card_2d[0]] + SUIT_SYMBOL[card_2d[1]]


def hand_str(cards):
    return ' '.join(card_str(c) for c in cards)


def board_str(board, start, end):
    return ' '.join(card_str(board[i]) for i in range(start, end))


def render_state(env):
    round_names = {1: 'FLOP', 2: 'TURN', 3: 'RIVER'}
    print(f"\n  {'─' * 50}")
    print(f"  Round: {round_names.get(env.current_round, '?')}    Pot: {env.main_pot}")
    print(f"  Board 1:  {board_str(env.board, 0, 5)}")
    print(f"  Board 2:  {board_str(env.board, 5, 10)}")
    print(f"  P0: {hand_str(env.seats[0].hand)}  Stack: {env.seats[0].stack}")
    print(f"  P1: {hand_str(env.seats[1].hand)}  Stack: {env.seats[1].stack}")
    print(f"  {'─' * 50}")


def self_play():
    agent_base = os.path.expanduser('~/poker_ai_data/eval_agent/')
    agents = []
    for name in os.listdir(agent_base):
        agent_dir = os.path.join(agent_base, name)
        if os.path.isdir(agent_dir):
            for iter_dir in sorted(os.listdir(agent_dir)):
                pkl = os.path.join(agent_dir, iter_dir, 'eval_agentSINGLE.pkl')
                if os.path.exists(pkl):
                    agents.append((name, iter_dir, pkl))

    if not agents:
        print("No trained agents found.")
        return

    print("\nAvailable agents:")
    for i, (name, iteration, _) in enumerate(agents):
        print(f"  {i}: {name} (iter {iteration})")

    if len(agents) == 1:
        choice = 0
    else:
        choice = int(input(f"\nSelect agent [0-{len(agents)-1}]: "))

    name, iteration, pkl_path = agents[choice]
    print(f"\nLoading {name} iter {iteration}...")

    agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=pkl_path)
    env_cls = agent.env_bldr.env_cls
    env_args = agent.env_bldr.env_args
    env = env_cls(env_args=env_args, is_evaluating=True, lut_holder=env_cls.get_lut_holder())

    delay = float(input("\nDelay between actions (seconds, e.g. 1.0): ") or "1.0")
    n_hands = int(input("Number of hands to play (e.g. 20): ") or "20")

    p0_total = 0
    p1_total = 0

    stats = {'p0_folds': 0, 'p1_folds': 0, 'showdowns': 0,
             'p0_bets': 0, 'p1_bets': 0, 'p0_checks': 0, 'p1_checks': 0,
             'p0_scoops': 0, 'p1_scoops': 0, 'splits': 0}

    for hand_num in range(1, n_hands + 1):
        print(f"\n{'=' * 60}")
        print(f"  HAND #{hand_num}    P0: {p0_total:+d}  P1: {p1_total:+d}")
        print(f"{'=' * 60}")

        obs, _, done, _ = env.reset()
        agent.reset(deck_state_dict=env.cards_state_dict())

        render_state(env)
        time.sleep(delay)

        actions_this_hand = []

        while not done:
            p_id = env.current_player.seat_id
            a_idx_raw, _ = agent.get_action(step_env=True, need_probs=False)
            action_tuple = env._get_env_adjusted_action_formulation(a_idx_raw)
            action_tuple = env._get_fixed_action(action_tuple)

            action_name = ACTION_NAMES[a_idx_raw] if a_idx_raw < len(ACTION_NAMES) else f'ACTION_{a_idx_raw}'
            print(f"  P{p_id}: {action_name}")
            actions_this_hand.append((p_id, action_name))

            # Track stats
            if a_idx_raw == 0:
                if p_id == 0: stats['p0_folds'] += 1
                else: stats['p1_folds'] += 1
            elif a_idx_raw == 1:
                if p_id == 0: stats['p0_checks'] += 1
                else: stats['p1_checks'] += 1
            else:
                if p_id == 0: stats['p0_bets'] += 1
                else: stats['p1_bets'] += 1

            obs, rews, done, info = env._step(processed_action=action_tuple)

            if not done and env.current_round != (actions_this_hand[-1] if actions_this_hand else None):
                render_state(env)

            time.sleep(delay)

        # Results
        render_state(env)
        r0 = int(np.rint(rews[0] * env.REWARD_SCALAR))
        r1 = int(np.rint(rews[1] * env.REWARD_SCALAR))
        p0_total += r0
        p1_total += r1

        if r0 > 0:
            print(f"\n  >>> P0 wins {r0} chips <<<")
        elif r1 > 0:
            print(f"\n  >>> P1 wins {r1} chips <<<")
        else:
            print(f"\n  >>> Push <<<")

        # Showdown stats
        if env.seats[0].hand_rank_board1 > 0:
            stats['showdowns'] += 1
            b1_p0 = env.seats[0].hand_rank_board1
            b1_p1 = env.seats[1].hand_rank_board1
            b2_p0 = env.seats[0].hand_rank_board2
            b2_p1 = env.seats[1].hand_rank_board2
            b1_winner = 'P0' if b1_p0 > b1_p1 else ('P1' if b1_p1 > b1_p0 else 'TIE')
            b2_winner = 'P0' if b2_p0 > b2_p1 else ('P1' if b2_p1 > b2_p0 else 'TIE')
            print(f"  Board 1: {b1_winner} wins (P0={b1_p0} P1={b1_p1})")
            print(f"  Board 2: {b2_winner} wins (P0={b2_p0} P1={b2_p1})")

            if b1_winner == b2_winner and b1_winner != 'TIE':
                if b1_winner == 'P0': stats['p0_scoops'] += 1
                else: stats['p1_scoops'] += 1
            else:
                stats['splits'] += 1

        time.sleep(delay)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY ({n_hands} hands)")
    print(f"{'=' * 60}")
    print(f"  P0 total: {p0_total:+d}    P1 total: {p1_total:+d}")
    print(f"\n  Action stats:")
    print(f"    P0 bets: {stats['p0_bets']}  checks: {stats['p0_checks']}  folds: {stats['p0_folds']}")
    print(f"    P1 bets: {stats['p1_bets']}  checks: {stats['p1_checks']}  folds: {stats['p1_folds']}")
    print(f"\n  Showdown stats ({stats['showdowns']} showdowns):")
    print(f"    P0 scoops: {stats['p0_scoops']}  P1 scoops: {stats['p1_scoops']}  Splits: {stats['splits']}")


if __name__ == '__main__':
    self_play()
