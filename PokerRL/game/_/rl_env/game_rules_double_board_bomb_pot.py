from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules_plo import PLORules


class DoubleBoardBombPotPLORules(PLORules):
    """
    Rules for Double Board Bomb Pot PLO.
    - Two independent 5-card community boards
    - No preflop betting round (bomb pot: everyone posts ante, action starts on flop)
    - Pot splits 50/50 between board 1 winner and board 2 winner
    """

    N_BOARDS = 2
    N_TOTAL_BOARD_CARDS = 10  # 5 per board

    # No preflop round in bomb pots
    ALL_ROUNDS_LIST = [Poker.FLOP, Poker.TURN, Poker.RIVER]

    ROUND_BEFORE = {
        Poker.FLOP: Poker.FLOP,
        Poker.TURN: Poker.FLOP,
        Poker.RIVER: Poker.TURN,
    }
    ROUND_AFTER = {
        Poker.FLOP: Poker.TURN,
        Poker.TURN: Poker.RIVER,
        Poker.RIVER: None,
    }

    STRING = "DOUBLE_BOARD_BOMB_POT_PLO_RULES"

    @classmethod
    def get_lut_holder(cls):
        from PokerRL.game._.look_up_table import LutHolderPLO
        return LutHolderPLO(cls)
