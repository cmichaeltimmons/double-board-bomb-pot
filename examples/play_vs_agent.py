"""
Play against the Double Board Bomb Pot PLO agent interactively.
You play as Player 0, the AI plays as Player 1.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.Poker import Poker


RANK_STR = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A'}
SUIT_STR = {0:'h', 1:'d', 2:'s', 3:'c'}
SUIT_SYMBOL = {0:'\u2665', 1:'\u2666', 2:'\u2660', 3:'\u2663'}  # hearts, diamonds, spades, clubs

ACTION_NAMES = ['FOLD', 'CHECK/CALL', 'BET 33% POT', 'BET 50% POT', 'BET POT']


def card_str(card_2d):
    if card_2d[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
        return '??'
    return RANK_STR[card_2d[0]] + SUIT_SYMBOL[card_2d[1]]


def hand_str(cards):
    return ' '.join(card_str(c) for c in cards)


def board_str(board, start, end):
    cards = []
    for i in range(start, end):
        if board[i][0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
            cards.append('--')
        else:
            cards.append(card_str(board[i]))
    return ' '.join(cards)


def render_state(env, your_hand, show_ai_hand=False):
    print()
    print('=' * 60)
    round_names = {1: 'FLOP', 2: 'TURN', 3: 'RIVER'}
    print(f"  Round: {round_names.get(env.current_round, '?')}    Pot: {env.main_pot}")
    print()
    print(f"  Board 1:  {board_str(env.board, 0, 5)}")
    print(f"  Board 2:  {board_str(env.board, 5, 10)}")
    print()
    print(f"  Your hand (P0):  {hand_str(your_hand)}    Stack: {env.seats[0].stack}")
    if show_ai_hand:
        print(f"  AI hand   (P1):  {hand_str(env.seats[1].hand)}    Stack: {env.seats[1].stack}")
    else:
        print(f"  AI hand   (P1):  ?? ?? ?? ??    Stack: {env.seats[1].stack}")
    print('=' * 60)


def play():
    # Find the most recent agent
    agent_base = os.path.expanduser('~/poker_ai_data/eval_agent/')
    if not os.path.exists(agent_base):
        print("No trained agents found. Run training first.")
        return

    # List available agents
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

    n_actions = agent.env_bldr.N_ACTIONS
    bet_fracs = env_args.bet_sizes_list_as_frac_of_pot

    print(f"\nGame: Double Board Bomb Pot PLO")
    print(f"Ante: {env.ANTE}  Stack: {env.DEFAULT_STACK_SIZE}")
    print(f"\nActions:")
    for i in range(n_actions):
        print(f"  {i}: {ACTION_NAMES[i] if i < len(ACTION_NAMES) else f'BET {bet_fracs[i-2]*100:.0f}%'}")

    winnings = 0
    hand_num = 0

    while True:
        hand_num += 1
        print(f"\n{'*' * 60}")
        print(f"  HAND #{hand_num}    (Your total: {winnings:+d} chips)")
        print(f"{'*' * 60}")

        # Reset
        obs, _, done, _ = env.reset()
        agent.reset(deck_state_dict=env.cards_state_dict())

        your_hand = env.seats[0].hand

        render_state(env, your_hand)

        while not done:
            p_id = env.current_player.seat_id
            legal = env.get_legal_actions()

            if p_id == 0:
                # Human turn
                legal_strs = [f"{a}={ACTION_NAMES[a]}" for a in legal]
                print(f"\n  Legal: {', '.join(legal_strs)}")

                while True:
                    try:
                        action = int(input("  Your action> "))
                        if action in legal:
                            break
                        print(f"  Illegal! Choose from: {legal}")
                    except (ValueError, EOFError):
                        print(f"  Enter a number from: {legal}")

                # Convert to processed action for the env
                action_tuple = env._get_env_adjusted_action_formulation(action)
                action_tuple = env._get_fixed_action(action_tuple)

                # Notify agent of human action
                agent.notify_of_processed_tuple_action(action_he_did=action_tuple, p_id_acted=0)

            else:
                # AI turn
                a_idx_raw, _ = agent.get_action(step_env=True, need_probs=False)
                action_tuple = env._get_env_adjusted_action_formulation(a_idx_raw)
                action_tuple = env._get_fixed_action(action_tuple)
                print(f"\n  AI plays: {ACTION_NAMES[a_idx_raw] if a_idx_raw < len(ACTION_NAMES) else a_idx_raw}")

            obs, rews, done, info = env._step(processed_action=action_tuple)

            if not done:
                render_state(env, your_hand)

        # Showdown
        print()
        render_state(env, your_hand, show_ai_hand=True)

        reward = int(np.rint(rews[0] * env.REWARD_SCALAR))
        winnings += reward

        if reward > 0:
            print(f"\n  >>> YOU WIN {reward} chips! <<<")
        elif reward < 0:
            print(f"\n  >>> You lose {-reward} chips <<<")
        else:
            print(f"\n  >>> Push (0 chips) <<<")

        # Show hand ranks if showdown happened
        if env.seats[0].hand_rank_board1 > 0:
            print(f"  Board 1: You={env.seats[0].hand_rank_board1}  AI={env.seats[1].hand_rank_board1}")
            print(f"  Board 2: You={env.seats[0].hand_rank_board2}  AI={env.seats[1].hand_rank_board2}")

        try:
            input("\n  Press Enter for next hand (Ctrl+C to quit)...")
        except (KeyboardInterrupt, EOFError):
            print(f"\n\nFinal results: {winnings:+d} chips over {hand_num} hands")
            break


if __name__ == '__main__':
    play()
