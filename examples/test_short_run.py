"""Short test run of Double Board Bomb Pot PLO training - 2 iterations, HU."""
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch

from PokerRL.game.games import DoubleBoardBombPotPLO
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.game import bet_sets
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import VanillaEnvBuilder

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
                      name="DBBP_test_short",
                      nn_type="dense_residual",

                      DISTRIBUTED=False,
                      CLUSTER=False,
                      n_learner_actor_workers=1,

                      max_buffer_size_adv=50000,
                      export_each_net=False,
                      checkpoint_freq=999,
                      eval_agent_export_freq=999,

                      n_actions_traverser_samples=3,
                      n_traversals_per_iter=500,
                      n_batches_adv_training=50,

                      use_pre_layers_adv=True,
                      n_cards_state_units_adv=64,
                      n_merge_and_table_layer_units_adv=32,
                      n_units_final_adv=32,
                      lr_patience_adv=99999,
                      lr_adv=0.001,

                      mini_batch_size_adv=256,
                      init_adv_model="last",

                      game_cls=DoubleBoardBombPotPLO,
                      env_bldr_cls=VanillaEnvBuilder,
                      agent_bet_set=bet_sets.PL_THIRD_HALF_POT,
                      n_seats=2,
                      start_chips=10000,

                      eval_modes_of_algo=(
                          EvalAgentDeepCFR.EVAL_MODE_SINGLE,
                      ),

                      use_simplified_headsup_obs=True,

                      log_verbose=True,
                      lbr_args=LBRArgs(lbr_bet_set=bet_sets.PL_THIRD_HALF_POT,
                                       n_lbr_hands_per_seat=1,
                                       lbr_check_to_round=Poker.TURN,
                                       n_parallel_lbr_workers=1,
                                       use_gpu_for_batch_eval=False,
                                       DISTRIBUTED=False,
                                       ),
                  ),
                  eval_methods={},
                  n_iterations=2)

    ctrl.run()
