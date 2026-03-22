import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
import random
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
                      name="DBBP_PLO_HU_fixed_flop",
                      nn_type="dense_residual",

                      DISTRIBUTED=False,
                      CLUSTER=False,
                      n_learner_actor_workers=1,

                      max_buffer_size_adv=3000000,
                      export_each_net=False,
                      checkpoint_freq=20,
                      eval_agent_export_freq=5,

                      n_actions_traverser_samples=4,
                      n_traversals_per_iter=50000,
                      n_batches_adv_training=1500,

                      use_pre_layers_adv=True,
                      n_cards_state_units_adv=192,
                      n_merge_and_table_layer_units_adv=64,
                      n_units_final_adv=64,
                      lr_patience_adv=350,
                      lr_adv=0.004,

                      mini_batch_size_adv=5000,
                      init_adv_model="last",

                      device_inference="cpu",
                      device_training="cuda",
                      device_parameter_server="cpu",

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
                  n_iterations=500,
                  iteration_to_import=60,
                  name_to_import="DBBP_PLO_HU_fixed_flop")

    ctrl.run()
