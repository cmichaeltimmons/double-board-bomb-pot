"""
Claude's strategy plays against the trained agent.
Uses hand strength evaluation to make decisions.
Default 100k hands, prints summary every 1k hands.
Usage: python claude_vs_agent.py [n_hands]
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.Poker import Poker
from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval

ACTION_NAMES = ['FOLD', 'CHECK/CALL', 'BET 33%', 'BET 50%', 'BET POT']
RANK_STR = '23456789TJQKA'
SUIT_SYMBOL = ['h', 'd', 's', 'c']

hand_evaluator = CppHandeval()


def card_str(card_2d):
    if card_2d[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
        return '--'
    return RANK_STR[card_2d[0]] + SUIT_SYMBOL[card_2d[1]]


def hand_str(cards):
    return ' '.join(card_str(c) for c in cards)


def board_str(board, start, end):
    return ' '.join(card_str(board[i]) for i in range(start, end))


def get_hand_strength(hand_2d, board_2d):
    """Get PLO hand rank (higher = better). Returns -1 if board not dealt yet."""
    if board_2d[0][0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
        return -1
    dealt = []
    for c in board_2d:
        if c[0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
            dealt.append(c)
    if len(dealt) < 3:
        return -1
    board_5 = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    for i, c in enumerate(dealt):
        board_5[i] = c
    return hand_evaluator.get_hand_rank_52_plo(hand_2d, board_5)


def assess_both_boards(hand_2d, board):
    """Assess hand on both boards. Returns (rank1, rank2)."""
    board1 = board[0:5]
    board2 = board[5:10]
    r1 = get_hand_strength(hand_2d, board1)
    r2 = get_hand_strength(hand_2d, board2)
    return r1, r2


def choose_action(hand_2d, board, legal_actions, pot, my_stack, opp_stack, current_round):
    """
    Simple but reasonable strategy:
    - Assess hand strength on both boards
    - Strong on both = bet big
    - Strong on one = bet small or call
    - Weak on both = check or fold to big bets
    """
    r1, r2 = assess_both_boards(hand_2d, board)

    if r1 < 0 and r2 < 0:
        return 1 if 1 in legal_actions else 0

    # Categorize hand ranks
    # In PLO, typical hand ranks: <500k = weak, 500k-650k = marginal, 650k-900k = good, >900k = great
    def strength(r):
        if r < 0: return 0
        if r > 900000: return 3   # great (trips+, straights, flushes)
        if r > 650000: return 2   # good (two pair, strong pair)
        if r > 500000: return 1   # marginal (pair, weak two pair)
        return 0                   # weak (high card, bottom pair)

    s1 = strength(r1)
    s2 = strength(r2)
    total = s1 + s2  # 0-6

    # If fold is legal, there's a bet to face
    facing_bet = 0 in legal_actions

    # How much of our stack is at risk
    stack_commitment = 1.0 - (my_stack / 10000.0) if my_stack < 10000 else 0.0

    if total >= 5:
        # Monster on both boards - bet big / raise
        if 4 in legal_actions:
            return 4
        if 3 in legal_actions:
            return 3
        return max(legal_actions)

    elif total >= 4:
        # Strong on both boards - bet or raise
        if 3 in legal_actions:
            return 3
        if 2 in legal_actions:
            return 2
        return 1 if 1 in legal_actions else 0

    elif total >= 3:
        # Good on one board, ok on the other
        if not facing_bet:
            if 2 in legal_actions:
                return 2  # small bet
            return 1  # check
        else:
            if stack_commitment < 0.5:
                return 1 if 1 in legal_actions else 0
            else:
                if total >= 3:
                    return 1 if 1 in legal_actions else 0
                return 0  # fold

    elif total >= 2:
        # Marginal - check or call small bets
        if not facing_bet:
            return 1  # check
        else:
            if stack_commitment < 0.3:
                return 1 if 1 in legal_actions else 0  # call if cheap
            return 0  # fold to big bets

    elif total == 1:
        # Weak - check or fold
        if not facing_bet:
            return 1  # check
        if stack_commitment < 0.1:
            return 1 if 1 in legal_actions else 0  # call if very cheap
        return 0  # fold

    else:
        # Air - check or fold
        if not facing_bet:
            return 1  # check
        return 0  # fold


def play_match(n_hands=100000, report_interval=1000):
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

    total_winnings = 0
    interval_winnings = 0
    stats = {'my_folds': 0, 'ai_folds': 0, 'my_bets': 0, 'ai_bets': 0,
             'my_checks': 0, 'ai_checks': 0, 'showdowns': 0, 'scoops_me': 0,
             'scoops_ai': 0, 'splits': 0}

    print(f"\n{'=' * 80}")
    print(f"  CLAUDE STRATEGY vs {name} iter {iteration}  |  {n_hands:,} hands, report every {report_interval:,}")
    print(f"  Positions alternate each hand (even=P0, odd=P1)")
    print(f"{'=' * 80}")
    print(f"  {'Hands':>8}  {'Interval':>10}  {'Total':>10}  {'mA/hand':>10}  {'Showdowns':>10}  {'Scoops(C/A)':>12}  {'Splits':>7}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*7}")

    for hand_num in range(1, n_hands + 1):
        # Alternate positions each hand: even hands Claude=P0, odd hands Claude=P1
        claude_seat = hand_num % 2  # 0 or 1
        agent_seat = 1 - claude_seat

        obs, _, done, _ = env.reset()
        agent.reset(deck_state_dict=env.cards_state_dict())
        my_hand = env.seats[claude_seat].hand

        while not done:
            p_id = env.current_player.seat_id
            legal = env.get_legal_actions()

            if p_id == claude_seat:
                action = choose_action(
                    my_hand, env.board, legal, env.main_pot,
                    env.seats[claude_seat].stack, env.seats[agent_seat].stack, env.current_round
                )
                if action == 0: stats['my_folds'] += 1
                elif action == 1: stats['my_checks'] += 1
                else: stats['my_bets'] += 1

                action_tuple = env._get_env_adjusted_action_formulation(action)
                action_tuple = env._get_fixed_action(action_tuple)
                agent.notify_of_processed_tuple_action(action_he_did=action_tuple, p_id_acted=claude_seat)
            else:
                a_idx_raw, _ = agent.get_action(step_env=True, need_probs=False)
                action = a_idx_raw
                if action == 0: stats['ai_folds'] += 1
                elif action == 1: stats['ai_checks'] += 1
                else: stats['ai_bets'] += 1

                action_tuple = env._get_env_adjusted_action_formulation(action)
                action_tuple = env._get_fixed_action(action_tuple)

            obs, rews, done, info = env._step(processed_action=action_tuple)

        reward = int(np.rint(rews[claude_seat] * env.REWARD_SCALAR))
        total_winnings += reward
        interval_winnings += reward

        if hasattr(env.seats[claude_seat], 'hand_rank_board1') and env.seats[claude_seat].hand_rank_board1 > 0:
            stats['showdowns'] += 1
            b1_me = env.seats[claude_seat].hand_rank_board1
            b1_ai = env.seats[agent_seat].hand_rank_board1
            b2_me = env.seats[claude_seat].hand_rank_board2
            b2_ai = env.seats[agent_seat].hand_rank_board2
            if b1_me > b1_ai and b2_me > b2_ai: stats['scoops_me'] += 1
            elif b1_ai > b1_me and b2_ai > b2_me: stats['scoops_ai'] += 1
            else: stats['splits'] += 1

        if hand_num % report_interval == 0:
            ma_per_hand = (total_winnings / hand_num) * env.EV_NORMALIZER
            print(f"  {hand_num:>8,}  {interval_winnings:>+10,}  {total_winnings:>+10,}  {ma_per_hand:>+10.1f}  {stats['showdowns']:>10,}  {stats['scoops_me']:>5,}/{stats['scoops_ai']:<5,}  {stats['splits']:>7,}")
            interval_winnings = 0

    # Final summary
    ma_per_hand = (total_winnings / n_hands) * env.EV_NORMALIZER
    print(f"\n{'=' * 80}")
    print(f"  FINAL RESULTS  |  {n_hands:,} hands vs {name} iter {iteration}")
    print(f"{'=' * 80}")
    print(f"  Total: {total_winnings:+,} chips  |  {ma_per_hand:+.1f} mA/hand")
    print(f"  Claude - bets: {stats['my_bets']:,}  checks: {stats['my_checks']:,}  folds: {stats['my_folds']:,}")
    print(f"  Agent  - bets: {stats['ai_bets']:,}  checks: {stats['ai_checks']:,}  folds: {stats['ai_folds']:,}")
    print(f"  Showdowns: {stats['showdowns']:,}  |  Claude scoops: {stats['scoops_me']:,}  Agent scoops: {stats['scoops_ai']:,}  Splits: {stats['splits']:,}")
    print(f"\n  {'CLAUDE STRATEGY WINS!' if total_winnings > 0 else 'AGENT WINS!' if total_winnings < 0 else 'DRAW!'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_hands', nargs='?', type=int, default=100000, help='Number of hands (default: 100000)')
    parser.add_argument('--interval', type=int, default=1000, help='Report every N hands (default: 1000)')
    args = parser.parse_args()
    play_match(n_hands=args.n_hands, report_interval=args.interval)
