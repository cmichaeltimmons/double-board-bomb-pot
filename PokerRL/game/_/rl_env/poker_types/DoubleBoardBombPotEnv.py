import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.poker_types.DiscretizedPokerEnv import DiscretizedPokerEnv


class DoubleBoardBombPotEnv(DiscretizedPokerEnv):
    """
    Double Board Bomb Pot environment.
    - Two independent 5-card boards dealt simultaneously
    - No preflop betting (bomb pot: everyone posts ante, action starts on flop)
    - Pot splits 50/50 between board 1 winner and board 2 winner

    Board layout in self.board (shape (10, 2)):
        indices 0-4: board 1 (flop[0:3], turn[3], river[4])
        indices 5-9: board 2 (flop[5:8], turn[8], river[9])
    """

    def __init__(self, env_args, lut_holder, is_evaluating, hh_logger=None):
        super().__init__(env_args=env_args, lut_holder=lut_holder,
                         is_evaluating=is_evaluating, hh_logger=hh_logger)

    # _______________________________ Board Dealing _______________________________

    def _deal_flop(self):
        self.board[:3] = self.deck.draw(3)       # board 1 flop
        self.board[5:8] = self.deck.draw(3)      # board 2 flop

    def _deal_turn(self):
        self.board[3:4] = self.deck.draw(1)      # board 1 turn
        self.board[8:9] = self.deck.draw(1)      # board 2 turn

    def _deal_river(self):
        self.board[4:5] = self.deck.draw(1)      # board 1 river
        self.board[9:10] = self.deck.draw(1)     # board 2 river

    # _______________________________ Board State Encoding _______________________________

    def _get_board_state(self):
        """
        Override to use 'continue' instead of 'break' on undealt cards.
        The base implementation breaks on CARD_NOT_DEALT_TOKEN_1D, which would
        skip board 2 entirely since board 1 has undealt slots between the boards.
        """
        K = (self.N_RANKS + self.N_SUITS)
        _board_space = [0] * (self.N_TOTAL_BOARD_CARDS * K)
        for i, card in enumerate(self.board.tolist()):
            if card[0] == Poker.CARD_NOT_DEALT_TOKEN_1D:
                continue
            D = K * i
            _board_space[card[0] + D] = 1
            if self.SUITS_MATTER:
                _board_space[card[1] + D + self.N_RANKS] = 1
        return _board_space

    # _______________________________ Bomb Pot Reset _______________________________

    def reset(self, deck_state_dict=None):
        """
        Bomb pot reset: everyone posts ante, no blinds, action starts on the flop.
        """
        self.n_raises_this_round = 0

        # reset table
        self.side_pots = [0] * self.N_SEATS
        self.main_pot = 0
        self.board = self._get_new_board()
        self.last_action = [None, None, None]
        self.current_round = Poker.FLOP  # skip preflop
        self.capped_raise.reset()
        self.last_raiser = None
        self.n_actions_this_episode = 0

        # players reset
        for p in self.seats:
            p.reset()

        # reset deck
        self.deck.reset()

        # start hand in logger if presented
        if self._hh_logger is not None:
            players = []
            for p in range(self.N_SEATS):
                players.append(("Player" + str(p), self.seats[p].stack))
            self._hh_logger.start_hand(players, button_pos=0, sb_pos=0, bb_pos=1)

        # bomb pot: everyone posts ante, no blinds
        self._post_antes()
        self._put_current_bets_into_main_pot_and_side_pots()

        # deal hole cards and flop to both boards
        self._deal_hole_cards()
        self._deal_flop()

        self.current_player = self._get_first_to_act_post_flop()

        # optionally synchronize random variables from another env
        if deck_state_dict is not None:
            self.load_cards_state_dict(cards_state_dict=deck_state_dict)

        return self._get_current_step_returns(is_terminal=False, info=[False, None])

    # _______________________________ Min Raise Override _______________________________

    def _get_current_total_min_raise(self):
        """
        Override to use ANTE as minimum delta when BIG_BLIND is 0 (bomb pot).
        """
        min_delta = self.ANTE if self.BIG_BLIND == 0 else self.BIG_BLIND

        if self.N_SEATS == 2:
            _sorted_ascending = sorted([p.current_bet for p in self.seats])
            delta = max(_sorted_ascending[1] - _sorted_ascending[0], min_delta)
            return _sorted_ascending[1] + delta
        else:
            current_bets_sorted_descending = sorted([p.current_bet for p in self.seats], reverse=True)
            current_to_call_total = max(current_bets_sorted_descending)
            _largest_bet = current_bets_sorted_descending[0]

            for i in range(1, self.N_SEATS):
                if current_bets_sorted_descending[i] == _largest_bet:
                    continue
                delta_between_last_and_before_last = _largest_bet - current_bets_sorted_descending[i]
                delta = max(delta_between_last_and_before_last, min_delta)
                return current_to_call_total + delta

            return current_to_call_total + min_delta

    # _______________________________ Hand Evaluation _______________________________

    def _assign_hand_ranks_to_all_players(self):
        """
        Evaluate each player's hand against both boards independently.
        """
        board1 = self.board[:5]
        board2 = self.board[5:]
        for player in self.seats:
            player.hand_rank_board1 = self.get_hand_rank(hand_2d=player.hand, board_2d=board1)
            player.hand_rank_board2 = self.get_hand_rank(hand_2d=player.hand, board_2d=board2)
            # Set hand_rank to max of both for compatibility with code that reads it
            player.hand_rank = max(player.hand_rank_board1, player.hand_rank_board2)

    # _______________________________ Pot Distribution _______________________________

    def _payout_pots(self):
        """
        Split each pot 50/50 between board 1 winner and board 2 winner.
        """
        self._assign_hand_ranks_to_all_players()

        if self.N_SEATS == 2:
            self._payout_pots_hu()
        else:
            self._payout_pots_multi()

        if self._hh_logger is not None:
            self._hh_logger.show_down()

    def _payout_pots_hu(self):
        """Heads-up payout with double board 50/50 split."""
        half_pot = self.main_pot / 2

        for board_attr in ['hand_rank_board1', 'hand_rank_board2']:
            r0 = getattr(self.seats[0], board_attr)
            r1 = getattr(self.seats[1], board_attr)

            if r0 > r1:
                self.seats[0].award(half_pot)
            elif r0 < r1:
                self.seats[1].award(half_pot)
            else:
                self.seats[0].award(half_pot / 2)
                self.seats[1].award(half_pot / 2)

        self.main_pot = 0

    def _payout_pots_multi(self):
        """Multi-player payout with double board 50/50 split."""
        pots = np.array([self.main_pot] + self.side_pots)
        pot_ranks = np.arange(start=-1, stop=len(self.side_pots))
        pot_and_pot_ranks = np.array((pots, pot_ranks)).T

        for e in pot_and_pot_ranks:
            pot = e[0]
            rank = e[1]
            eligible_players = [p for p in self.seats if p.side_pot_rank >= rank and not p.folded_this_episode]

            if len(eligible_players) > 0:
                half_pot = pot / 2

                for board_attr in ['hand_rank_board1', 'hand_rank_board2']:
                    self._award_half_pot(half_pot, eligible_players, board_attr)

        self.side_pots = [0] * self.N_SEATS
        self.main_pot = 0

    def _award_half_pot(self, half_pot, eligible_players, board_attr):
        """Award half of a pot based on hand ranks on a specific board."""
        best_rank = max(getattr(p, board_attr) for p in eligible_players)
        winners = [p for p in eligible_players if getattr(p, board_attr) == best_rank]
        num_winners = len(winners)

        chips_per_winner = int(half_pot / num_winners)
        remainder = int(half_pot) % num_winners

        for p in winners:
            p.award(chips_per_winner)

        # distribute remainder randomly
        if remainder > 0:
            shuffled_idxs = np.arange(num_winners)
            np.random.shuffle(shuffled_idxs)
            for idx in shuffled_idxs[:remainder]:
                winners[idx].award(1)
