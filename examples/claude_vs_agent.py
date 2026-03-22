"""
Claude's strategy plays 20 hands against the trained agent.
Uses hand strength evaluation to make decisions.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.Poker import Poker
from PokerRL.game._.cpp_wrappers.CppHandeval import CppHandeval

RANK_STR = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A'}
SUIT_SYMBOL = {0:'\u2665', 1:'\u2666', 2:'\u2660', 3:'\u2663'}
ACTION_NAMES = ['FOLD', 'CHECK/CALL', 'BET 33%', 'BET 50%', 'BET POT']

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


def play_match():
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

    n_hands = 20
    winnings = 0
    stats = {'my_folds': 0, 'ai_folds': 0, 'my_bets': 0, 'ai_bets': 0,
             'my_checks': 0, 'ai_checks': 0, 'showdowns': 0, 'scoops_me': 0,
             'scoops_ai': 0, 'splits': 0}

    print(f"\n{'=' * 60}")
    print(f"  CLAUDE STRATEGY vs TRAINED AGENT ({n_hands} hands)")
    print(f"  Claude = P0 (in position), Agent = P1 (out of position)")
    print(f"{'=' * 60}")

    for hand_num in range(1, n_hands + 1):
        print(f"\n{'~'*60}")
        print(f"  HAND #{hand_num}    Running total: {winnings:+d}")
        print(f"{'~'*60}")

        obs, _, done, _ = env.reset()
        agent.reset(deck_state_dict=env.cards_state_dict())

        my_hand = env.seats[0].hand

        # Show initial state
        print(f"  Board 1: {board_str(env.board, 0, 5)}")
        print(f"  Board 2: {board_str(env.board, 5, 10)}")
        print(f"  My hand:  {hand_str(my_hand)}")
        print(f"  Pot: {env.main_pot}")

        while not done:
            p_id = env.current_player.seat_id
            legal = env.get_legal_actions()

            if p_id == 0:
                # Claude's turn
                action = choose_action(
                    my_hand, env.board, legal, env.main_pot,
                    env.seats[0].stack, env.seats[1].stack, env.current_round
                )
                action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else f'ACT_{action}'
                print(f"    Claude: {action_name}")

                if action == 0: stats['my_folds'] += 1
                elif action == 1: stats['my_checks'] += 1
                else: stats['my_bets'] += 1

                action_tuple = env._get_env_adjusted_action_formulation(action)
                action_tuple = env._get_fixed_action(action_tuple)
                agent.notify_of_processed_tuple_action(action_he_did=action_tuple, p_id_acted=0)
            else:
                # Agent's turn
                a_idx_raw, _ = agent.get_action(step_env=True, need_probs=False)
                action_name = ACTION_NAMES[a_idx_raw] if a_idx_raw < len(ACTION_NAMES) else f'ACT_{a_idx_raw}'
                print(f"    Agent:  {action_name}")

                if a_idx_raw == 0: stats['ai_folds'] += 1
                elif a_idx_raw == 1: stats['ai_checks'] += 1
                else: stats['ai_bets'] += 1

                action_tuple = env._get_env_adjusted_action_formulation(a_idx_raw)
                action_tuple = env._get_fixed_action(action_tuple)

            obs, rews, done, info = env._step(processed_action=action_tuple)

        # Results
        reward = int(np.rint(rews[0] * env.REWARD_SCALAR))
        winnings += reward

        print(f"  Board 1: {board_str(env.board, 0, 5)}")
        print(f"  Board 2: {board_str(env.board, 5, 10)}")
        print(f"  My hand:  {hand_str(my_hand)}")
        print(f"  AI hand:  {hand_str(env.seats[1].hand)}")

        if reward > 0:
            print(f"  >>> CLAUDE WINS {reward} chips <<<")
        elif reward < 0:
            print(f"  >>> Claude loses {-reward} chips <<<")
        else:
            print(f"  >>> Push <<<")

        if env.seats[0].hand_rank_board1 > 0:
            stats['showdowns'] += 1
            b1_me = env.seats[0].hand_rank_board1
            b1_ai = env.seats[1].hand_rank_board1
            b2_me = env.seats[0].hand_rank_board2
            b2_ai = env.seats[1].hand_rank_board2
            b1_w = 'Claude' if b1_me > b1_ai else ('Agent' if b1_ai > b1_me else 'Tie')
            b2_w = 'Claude' if b2_me > b2_ai else ('Agent' if b2_ai > b2_me else 'Tie')
            print(f"  Board 1: {b1_w}  Board 2: {b2_w}")

            if b1_w == b2_w == 'Claude': stats['scoops_me'] += 1
            elif b1_w == b2_w == 'Agent': stats['scoops_ai'] += 1
            else: stats['splits'] += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS ({n_hands} hands)")
    print(f"{'=' * 60}")
    print(f"  Claude total: {winnings:+d} chips")
    print(f"  Per hand: {winnings/n_hands:+.0f} chips/hand")
    print(f"\n  Action stats:")
    print(f"    Claude - bets: {stats['my_bets']}  checks: {stats['my_checks']}  folds: {stats['my_folds']}")
    print(f"    Agent  - bets: {stats['ai_bets']}  checks: {stats['ai_checks']}  folds: {stats['ai_folds']}")
    print(f"\n  Showdown stats ({stats['showdowns']} showdowns):")
    print(f"    Claude scoops: {stats['scoops_me']}  Agent scoops: {stats['scoops_ai']}  Splits: {stats['splits']}")

    if winnings > 0:
        print(f"\n  CLAUDE STRATEGY WINS!")
    elif winnings < 0:
        print(f"\n  AGENT WINS!")
    else:
        print(f"\n  DRAW!")


if __name__ == '__main__':
    play_match()
