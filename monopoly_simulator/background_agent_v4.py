from monopoly_simulator import action_choices
from monopoly_simulator import agent_helper_functions_v2,background_agent_v5_base
# log.
import logging
import random
import torch
from monopoly_simulator import rl_agent_helper
from monopoly_simulator import location
from monopoly_simulator.flag_config import flag_config_dict
from monopoly_simulator.rl_agent_helper import get_action_vector_mapping, get_action_vector5
from monopoly_simulator.bank import Bank


logger = logging.getLogger('monopoly_simulator.logging_info.background_agent')
UNSUCCESSFUL_LIMIT = 2
"""
All external decision_agent functions must have the exact signatures we have indicated in this document. Beyond
that, we impose no restrictions (you can make the decision agent as complex as you like (including maintaining state),
and we use good faith to ensure you do not manipulate the gameboard. We will have mechanisms to check for inadvertent
changes or inconsistencies that get introduced in the gameboard (due to any reason, including possible subtle errors
in the simulator itself) a short while later.

If you decision agent does maintain state, or some kind of global data structure, please be careful when assigning the
same decision agent (as we do) to each player. We do provide some basic state to you already via 'code' in the make_*_move
functions. Specifically, if code is 1 it means the 'previous' move selected by the player was successful,
and if -1 it means it was unsuccessful. code of -1 is usually returned when an allowable move is invoked with parameters
that preempt the action from happening e.g., the player may decide to mortgage property that is already mortgaged,
which will return the failure code of -1 when the game actually tries to mortgage the property in action_choices.

Be careful to note what each function is supposed to return in addition to adhering to the expected signature. The examples
here are good guides.

Your functions can be called whatever you like, but the keys in decision_agent_methods should not be changed. The
respective functions must adhere in their signatures to the examples here. The agent in this file is simple and rule-based,
 rather than adaptive but capable of taking good actions in a number of eventualities.
 We detail the logic behind each decision in a separate document. This is the agent that will serve as the 'background'
 agent for purposes of evaluation.

"""

import numpy as np

from gym import Env, error, spaces, utils
from gym.utils import seeding
from actor_critic_agents.SAC_Discrete import SAC_Discrete

from actor_critic_utilities.data_structures.Config import Config
import random
import torch

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

config = Config()
config.seed = 1
#config.environment = Bit_Flipping_Environment(4)

config.num_episodes_to_run = 2000
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False


config.hyperparameters = {


        "learning_rate": 0.05,
        "linear_hidden_units": [150, 30, 30, 30],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 25.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 10.0,
        "normalise_rewards": False,
        "automatically_tune_entropy_hyperparameter": True,
        "add_extra_noise": False,
        "min_steps_before_learning": 50,
        "do_evaluation_iterations": True,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.1,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.1,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "batch_size": 250,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 10,
        "learning_updates_per_learning_session": 10,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0

}

#agent_sac=SAC_Discrete(config)


def learn(agent_sac,experiences):
    """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""

    states, actions, rewards, next_states = rl_agent_helper.extract_tensors(experiences)

    qf1_loss, qf2_loss = agent_sac.calculate_critic_losses(states, actions, rewards, next_states, mask_batch=0)
    agent_sac.update_critic_parameters(qf1_loss, qf2_loss)

    policy_loss, log_pi = agent_sac.calculate_actor_loss(states)
   # print("bav4,129 ................................................policy loss:", policy_loss, qf1_loss, qf2_loss)
    if agent_sac.automatic_entropy_tuning:
        alpha_loss = agent_sac.calculate_entropy_tuning_loss(log_pi)
    else:
        alpha_loss = None
    agent_sac.update_actor_parameters(policy_loss, alpha_loss)



STATE_FUNCTION = rl_agent_helper.get_state_vector2
REWARD_FUNCTION = rl_agent_helper.get_reward_v5


def get_output_action(player, current_gameboard, allowable_moves):
    action_failed = True
    while action_failed:
        state_vector = STATE_FUNCTION(current_gameboard, player)
        action_idx = rl_agent_helper.select_action_v2(player.agent, state_vector, player.agent.policy_net)
        # print(action_idx.item())
        player.agent.state = state_vector
        # TODO: Right now pushing action_idx, regardless of the actual action taken - Change this to reflect do_nothing.
        # TODO: BETTER THING TO DO WOULD BE TO GET THE ORDERED LIST OF ACTIONS AND THEN ITERATE OVER IT.
        # Will need to take care of that in get_action_location ()
        player.agent.action = action_idx
        action_to_take, params, action_failed = get_action_location(action_idx.item(), player, allowable_moves,
                                                    current_gameboard)
    # print(action_failed)
    return action_to_take, params


def get_output_action_v3(player, current_gameboard, allowable_moves):
    # while action_failed:
    action_failed = True
    player.agent.state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
    action_idxes = rl_agent_helper.select_action_v3(player.agent, player.agent.policy_net)

    for action_idx in action_idxes:
        action_to_take, params, action_failed = get_action_location(action_idx, player, allowable_moves,
                                                                    current_gameboard)
        if action_failed:
            player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
            # player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
            # player.agent.next_state = player.agent.state.to(player.agent.device)
            # rl_agent_helper.push_to_memory(player.agent)
            # player.agent.failed_actions += 1
        else:
            # player.agent.successful_actions += 1
            player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
            break
    return action_to_take, params


def get_output_action_v4_old(player, current_gameboard, allowable_moves):
    # while action_failed:
    action_failed = True
    player.agent.state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
    action_idxes = rl_agent_helper.select_action_v4(player.agent, player.agent.policy_net, allowable_moves)
    # print ("action vecror1:", )
    # print("action vecror2:", get_action_vector5(current_gameboard))
    for action_idx in action_idxes:
        action_to_take, params, action_failed = get_action_vector_mapping(action_idx, player, allowable_moves,
                                                                          current_gameboard)
        if action_failed:
            # player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
            # player.agent.reward = torch.Tensor([-1]).to(player.agent.device)
            # player.agent.next_state = player.agent.state.to(player.agent.device)
            # rl_agent_helper.push_to_memory(player.agent)
            player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
            player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
            player.agent.next_state = player.agent.state.to(player.agent.device)
            rl_agent_helper.push_to_memory(player.agent)
            player.agent.failed_actions += 1
        else:
            player.agent.successful_actions += 1
            player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
            break
    return action_to_take, params

# def get_output_action_v4(player, current_gameboard, allowable_moves, code, move):
#
#     action_failed = True
#     player.agent.state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
#     #action_idxes = select_action_v3_sac(player.agent, move, player,current_gameboard, allowable_moves, code)
#     #print(action_idxes)
#     player.agent.current_step += 1
#     player.agent.episodic_step += 1
#     agent_sac.global_step_number +=1
#     action_idxes = list()
#     if agent_sac.global_step_number < agent_sac.hyperparameters["min_steps_before_learning"]:
#         act = background_agent_v5_base.decision_agent_methods[move](player, current_gameboard, allowable_moves, code)
#         action_idxes.append(act)
#         #print("ba_v4 220, step, Picking random action ", agent_sac.global_step_number, action_idxes)
#
#     else:
#         a= agent_sac.pick_action(eval_ep=False,state=player.agent.state)
#         player.agent.action = torch.LongTensor([[a]]).to(player.agent.device)
#         #print("b_a_v4,113- player.agent.action",  player.agent.action)
#         act = background_agent_v5_base.decision_agent_methods[move](player, current_gameboard, allowable_moves, code)
#         action_idxes.append(act)
#
#     if len(action_idxes) > 1:
#         for action_idx in action_idxes:
#             # print(action_idx)
#             action_to_take, params, action_failed = get_action_vector_mapping(action_idx, player, allowable_moves,current_gameboard)
#
#             if action_failed:
#                 player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
#                 player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
#                 player.agent.next_state = player.agent.state.to(player.agent.device)
#                 rl_agent_helper.push_to_memory(player.agent)
#                 player.agent.failed_actions += 1
#             else:
#                 # player.agent.successful_actions += 1
#                 player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
#                 break
#     else:
#         #action_to_take, params, action_failed = get_action_vector_mapping(10, player, allowable_moves,current_gameboard)
#         action_to_take,params = list(action_idxes[0])
#         #print("b_a_v4,action to take n params:", type(action_to_take))
#         player.agent.successful_actions += 1
#
#         if type(action_to_take) is list:
#             # print(action_to_take, params)
#             action_to_take = action_to_take[0]
#
#         ids = player.agent.action_map_r[action_to_take]
#         # print("IDs: ", ids)
#         if len(ids) == 1:
#             action_id = list(ids)
#             action_id = action_id[0]
#             # print(action_id)
#             player.agent.action = torch.LongTensor([[action_id]]).to(player.agent.device)
#             # player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
#             # player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
#             # rl_agent_helper.push_to_memory(player.agent)
#         else:
#             # print(params)
#             if type(params) is list:
#                 # print(ids)
#                 params = params[0]
#                 asset = params
#                 assets = asset["offer"]
#                 asset = list(assets["property_set_wanted"])
#                 if len(asset) == 0:
#                     asset = list(assets["property_set_offered"])
#                 asset = asset[0]
#                 # print(asset)
#                 # asset = list(asset["property_set_wanted"])
#                 if isinstance(asset, location.RailroadLocation):
#                     col = "Railroad"
#                 elif isinstance(asset, location.UtilityLocation):
#                     col = "Utility"
#                 else:
#                     # print(asset)
#                     col = asset.color
#                 # col = "None"
#             elif isinstance(params['asset'], location.RailroadLocation):
#                 col = "Railroad"
#             elif isinstance(params['asset'], location.UtilityLocation):
#                 col = "Utility"
#             else:
#                 col = params['asset'].color
#             # print(params)
#             acts = get_action_vector5(current_gameboard)
#             filter_act = {k:acts[k] for k in list(ids)}
#             # print(col, filter_act)
#             for key,value in filter_act.items():
#                 if value["loc_group"] == col:
#                     desired_act = key
#                     # print(value)
#                     break
#             # print(col, desired_act, filter_act)
#             # action_id = desired_act
#             player.agent.action = torch.LongTensor([[desired_act]]).to(player.agent.device)
#             # player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
#             # player.agent.next_state = player.agent.state.to(player.agent.device)
#             # rl_agent_helper.push_to_memory(player.agent)
#
#     return action_to_take, params

def get_output_action_v4(player, current_gameboard, allowable_moves, code, move):

    action_failed = True
    player.agent.state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
    #action_idxes = select_action_v3_sac(player.agent, move, player,current_gameboard, allowable_moves, code)
    #print(action_idxes)
    player.agent.current_step += 1
    player.agent.episodic_step += 1
    player.agent.agent_sac.global_step_number +=1
    action_idxes = list()
    if player.agent.agent_sac.global_step_number < player.agent.agent_sac.hyperparameters["min_steps_before_learning"]:
        act = background_agent_v5_base.decision_agent_methods[move](player, current_gameboard, allowable_moves, code)
        action_idxes.append(act)
        #print("ba_v4 220, step, Picking random action ", agent_sac.global_step_number, action_idxes)

    else:
        a= player.agent.agent_sac.pick_action(eval_ep=False,state=player.agent.state)
        player.agent.action = torch.LongTensor([[a]]).to(player.agent.device)
        #print("b_a_v4,113- player.agent.action",  player.agent.action)
        act = background_agent_v5_base.decision_agent_methods[move](player, current_gameboard, allowable_moves, code)
        action_idxes.append(act)

    if len(action_idxes) > 1:
        for action_idx in action_idxes:
            # print(action_idx)
            action_to_take, params, action_failed = get_action_vector_mapping(action_idx, player, allowable_moves,current_gameboard)

            if action_failed:
                player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
                player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
                player.agent.next_state = player.agent.state.to(player.agent.device)
                rl_agent_helper.push_to_memory(player.agent)
                player.agent.failed_actions += 1
            else:
                # player.agent.successful_actions += 1
                player.agent.action = torch.LongTensor([[action_idx]]).to(player.agent.device)
                break
    else:
        #action_to_take, params, action_failed = get_action_vector_mapping(10, player, allowable_moves,current_gameboard)
        action_to_take,params = list(action_idxes[0])
        #print("b_a_v4,action to take n params:", type(action_to_take))
        player.agent.successful_actions += 1

        if type(action_to_take) is list:
            # print(action_to_take, params)
            action_to_take = action_to_take[0]

        ids = player.agent.action_map_r[action_to_take]
        # print("IDs: ", ids)
        if len(ids) == 1:
            action_id = list(ids)
            action_id = action_id[0]
            # print(action_id)
            player.agent.action = torch.LongTensor([[action_id]]).to(player.agent.device)
            # player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
            # player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
            # rl_agent_helper.push_to_memory(player.agent)
        else:
            # print(params)
            if type(params) is list:
                # print(ids)
                params = params[0]
                asset = params
                assets = asset["offer"]
                asset = list(assets["property_set_wanted"])
                if len(asset) == 0:
                    asset = list(assets["property_set_offered"])
                asset = asset[0]
                # print(asset)
                # asset = list(asset["property_set_wanted"])
                if isinstance(asset, location.RailroadLocation):
                    col = "Railroad"
                elif isinstance(asset, location.UtilityLocation):
                    col = "Utility"
                else:
                    # print(asset)
                    col = asset.color
                # col = "None"
            elif isinstance(params['asset'], location.RailroadLocation):
                col = "Railroad"
            elif isinstance(params['asset'], location.UtilityLocation):
                col = "Utility"
            else:
                col = params['asset'].color
            # print(params)
            acts = get_action_vector5(current_gameboard)
            filter_act = {k:acts[k] for k in list(ids)}
            # print(col, filter_act)
            for key,value in filter_act.items():
                if value["loc_group"] == col:
                    desired_act = key
                    # print(value)
                    break
            # print(col, desired_act, filter_act)
            # action_id = desired_act
            player.agent.action = torch.LongTensor([[desired_act]]).to(player.agent.device)
            # player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
            # player.agent.next_state = player.agent.state.to(player.agent.device)
            # rl_agent_helper.push_to_memory(player.agent)

    return action_to_take, params



def get_owned_assets(loc_group, player):
    assets_owned = []
    for asset in player.assets:
        if loc_group == 'Railroad' and isinstance(asset, location.RailroadLocation):
            assets_owned.append(asset)
        elif loc_group == 'Utility' and isinstance(asset, location.UtilityLocation):
            assets_owned.append(asset)
        elif loc_group == asset.color:
            assets_owned.append(asset)
    return assets_owned


def get_action_location(action_idx, player, allowable_moves, current_gameboard):
    action_dict = player.agent.act_loc_map[action_idx]
    # LOGIC FOR ACTIONS TO TAKE
    if action_dict['loc_group'] is not None:
        player_owns = player_owns_loc_type(action_dict['loc_group'], player)
        # print(action_dict)
        params = dict()
        # current_location =
        params['player'] = player
        params['current_gameboard'] = current_gameboard
        # SELL ORDER:
        # Sell Hotel sell_house_hotel
        # Sell House sell_house_hotel
        # Mortgage Property mortgage_property
        # Sell Property sell_property
        if action_dict['action_type'] == 'sell' and player_owns:
            asset_list = get_owned_assets(action_dict['loc_group'], player)
            # print(asset_list)
            if action_dict['loc_group'] in player.full_color_sets_possessed \
                    and action_dict['loc_group'] not in ('Utility', 'Railroad'):
                # print('sell hotel/house')
                for asset_ in asset_list:
                    if asset_.num_hotels > 0:
                        params['asset'] = asset_
                        params['sell_house'] = False
                        params['sell_hotel'] = True
                        return action_choices.sell_house_hotel, params, False
                for asset_ in asset_list:
                    if asset_.num_houses > 0:
                        params['asset'] = asset_
                        params['sell_house'] = True
                        params['sell_hotel'] = False
                        return action_choices.sell_house_hotel, params, False
            elif action_choices.mortgage_property in allowable_moves:
                # print('mortgage_property')
                for asset_ in asset_list:
                    if not asset_.is_mortgaged:
                        params['asset'] = asset_
                        return action_choices.mortgage_property, params, False

            elif action_choices.sell_property in allowable_moves:
                # print('sell_property')
                # print(asset_list[0])
                params['asset'] = asset_list[0]
                return action_choices.sell_property, params, False
                # for asset in asset_list:
                #     print(asset)
                #     print('sell_property')

        # if a.action_type = 'sell'
        #   check if player owns anything in the location group -
        #   if yes,
        #       if they have a hotel on any of the properties
        #           if they do, sell it
        #       elif, check if they have a house on any of the properties
        #           if they do, sell it
        #       else if possible, mortgage one of the properties,
        #            if not possible, sell the property to either the bank or another player.

        # SPEND ORDER:
        # Add Hotel improve_property
        # Add House improve_property
        # Free mortgage free_mortgage
        # Buy Property buy_property

        elif action_dict['action_type'] == 'spend' and player_owns and action_choices.improve_property in allowable_moves:
            asset_list = get_owned_assets(action_dict['loc_group'], player)
            # print('improve_property', asset_list)
            if action_dict['loc_group'] not in ('Utility', 'Railroad'):
                for asset_ in asset_list:
                    if asset_.num_houses == 4:
                        params['asset'] = asset_
                        params['add_house'] = False
                        params['add_hotel'] = True
                        return action_choices.improve_property, params, False
                for asset_ in asset_list:
                    if not asset_.is_mortgaged:
                        params['asset'] = asset_
                        params['add_house'] = True
                        params['add_hotel'] = False
                        return action_choices.improve_property, params, False
            for asset_ in asset_list:
                if asset_.is_mortgaged:
                    params['asset'] = asset_
                    return action_choices.free_mortgage, params, False

        elif action_dict['action_type'] == 'spend' and action_choices.buy_property in allowable_moves:
            # print('buy_property')
            params = dict()
            # current_location =
            params['player'] = player
            params['asset'] = current_gameboard['location_sequence'][player.current_position]
            params['current_gameboard'] = current_gameboard
            return action_choices.buy_property, params, False
        # if a.action_type = 'spend'
        #   check if player owns anything in the location group -
        #   if yes,
        #       check if a hotel can be added to property
        # If multiple properties owned in the group - in an order check if you can build a hotel on it
        #           if yes, improve property
        #       elif, check if house can be added to property
        #           if yes, improve property
        #       elif property can be freed from mortgage ?
        #           yes - free mortgage
        #   if no, then check if the agent can buy the property - if yes, buy it.

        # DO NOTHING ORDER:
        # if skip_turn() in allowable moves, skip the turn
        # else concluded_actions()
        if action_dict['action_type'] != 'do nothing':
            # push to memory negative reward and no change in state

            return None, None, True
            # TODO - What could we do HERE? - done - let's see how it does :)
            # The problem is, even though it was not possible to perform the action that the network/random had as an
            # output, we're forcing the agent to take an action to conclude or skip
            # The agent will not learn this ? because the action output and the action taken are different
            # One way to mitigate this is to again call the action again?
            # print('FORCED SKIP/CONCLUDE')
            # ANS: Push it to memory with a negative reward and get another action -
            # keep getting actions and pushing till there is a valid action!
    elif action_dict['action_type'] == 'do_nothing':
        if action_choices.skip_turn in allowable_moves:
            return action_choices.skip_turn, dict(), False
        else:
            return action_choices.concluded_actions, dict(), False


def player_owns_loc_type(loc_group, player):
    # if len(player.assets) == 0:
    #     return False
    # else:
    for asset in player.assets:
        if loc_group == 'Railroad' and isinstance(asset, location.RailroadLocation):
            # print('owns railroad')
            return True
        elif loc_group == 'Utility' and isinstance(asset, location.UtilityLocation):
            # print('owns utility')
            return True
        elif loc_group == asset.color:
            # print('owns', asset.color)
            return True
    return False


OUTPUT_ACTION = get_output_action_v4


def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    """
    Many actions are possible in pre_roll.
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
    will always be a subset of the action choices for pre_die_roll in the game schema. Your returned action choice must be from
    allowable_moves; we will check for this when you return.
    :param code: See the preamble of this file for an explanation of this code
    :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
    parameters that will be passed into the function representing that action when it is executed.
    The dictionary must exactly contain the keys and expected value types expected by that action in
    action_choices
    """
    # Have the jail logic
    # Check if code = -1, then the last action was not valid - make sure you do not make the same action.
    # for p in current_gameboard['players']:
    #     if 'phase_game' not in p.agent._agent_memory:
    #         p.agent._agent_memory['phase_game'] = 0
    #         p.agent._agent_memory['count_unsuccessful_tries'] = 0
    #
    # if player.agent._agent_memory['phase_game'] != 0:
    #     player.agent._agent_memory['phase_game'] = 0
    #     for p in current_gameboard['players']:
    #         if p.status != 'lost':
    #             p.agent._agent_memory['count_unsuccessful_tries'] = 0
    #
    # if code == flag_config_dict['failure_code']:
    #     player.agent._agent_memory['count_unsuccessful_tries'] += 1
    #     if player.agent.action is not None:
    #         player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
    #         player.agent.next_state = player.agent.state.to(player.agent.device)
    #         rl_agent_helper.push_to_memory(player.agent)
    #         player.agent.failed_actions += 1
    #     logger.debug(player.player_name + ' has executed an unsuccessful preroll action, incrementing unsuccessful_tries ' +
    #                                       'counter to ' + str(player.agent._agent_memory['count_unsuccessful_tries']))
    if player.agent.action is not None:
        player.agent.successful_actions += 1
        player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
        player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
        rl_agent_helper.push_to_memory(player.agent)

    # if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
    #     logger.debug(player.player_name + ' has reached preroll unsuccessful action limits.')
    #     if "skip_turn" in allowable_moves:
    #         logger.debug(player.player_name+ ': I am skipping turn since I have crossed unsuccessful limits.')
    #         player.agent._agent_memory['previous_action'] = "skip_turn"
    #         return ("skip_turn", dict())
    #     elif "concluded_actions" in allowable_moves:
    #         # player.agent._agent_memory['previous_action'] = action_choices.concluded_actions
    #         logger.debug(player.player_name+ ': I am concluding actions since I have crossed unsuccessful limits.')
    #         return ("concluded_actions", dict())
    #     else:
    #         logger.error("Exception")
    #         raise Exception
    return OUTPUT_ACTION(player, current_gameboard, allowable_moves, code, "make_pre_roll_move")


def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    """
    The agent is in the out-of-turn phase and must decide what to do (next).
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
    will always be a subset of the action choices for out_of_turn in the game schema. Your returned action choice must be from
    allowable_moves; we will check for this when you return.
    :param code: See the preamble of this file for an explanation of this code
    :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
    parameters that will be passed into the function representing that action when it is executed.
    The dictionary must exactly contain the keys and expected value types expected by that action in
    action_choices
    """
    # for p in current_gameboard['players']:
    #     if 'phase_game' not in p.agent._agent_memory:
    #         p.agent._agent_memory['phase_game'] = 1
    #         p.agent._agent_memory['count_unsuccessful_tries'] = 0
    #
    # if player.agent._agent_memory['phase_game'] != 1:
    #     player.agent._agent_memory['phase_game'] = 1
    #     player.agent._agent_memory['count_unsuccessful_tries'] = 0
    #
    # if isinstance(code, list):
    #     code_flag = 0
    #     for c in code:
    #         if c == flag_config_dict['failure_code']:
    #             code_flag = 1
    #             break
    #     if code_flag:
    #         player.agent._agent_memory['count_unsuccessful_tries'] += 1
    #         if player.agent.action is not None:
    #             player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
    #             player.agent.next_state = player.agent.state.to(player.agent.device)
    #             rl_agent_helper.push_to_memory(player.agent)
    #             player.agent.failed_actions += 1
    #         logger.debug(player.player_name + ' has executed an unsuccessful out of turn action, incrementing unsuccessful_tries ' +
    #                                       'counter to ' + str(player.agent._agent_memory['count_unsuccessful_tries']))
    # elif code == flag_config_dict['failure_code']:
    #     player.agent._agent_memory['count_unsuccessful_tries'] += 1
    #     if player.agent.action is not None:
    #         player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
    #         player.agent.next_state = player.agent.state.to(player.agent.device)
    #         rl_agent_helper.push_to_memory(player.agent)
    #         player.agent.failed_actions += 1
    #     logger.debug(player.player_name + ' has executed an unsuccessful out of turn action, incrementing unsuccessful_tries ' +
    #                                       'counter to ' + str(player.agent._agent_memory['count_unsuccessful_tries']))

    if player.agent.action is not None:
        player.agent.successful_actions += 1
        player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
        player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
        rl_agent_helper.push_to_memory(player.agent)

    # if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
    #     logger.debug(player.player_name + ' has reached out of turn unsuccessful action limits.')
    #     if "skip_turn" in allowable_moves:
    #         logger.debug(player.player_name+ ': I am skipping turn since I have crossed unsuccessful limits.')
    #         player.agent._agent_memory['previous_action'] = "skip_turn"
    #         return ("skip_turn", dict())
    #     elif "concluded_actions" in allowable_moves:
    #         # player.agent._agent_memory['previous_action'] = action_choices.concluded_actions
    #         logger.debug(player.player_name+ ': I am concluding actions since I have crossed unsuccessful limits.')
    #         return ("concluded_actions", dict())
    #     else:
    #         logger.error("Exception")
    #         raise Exception
    return OUTPUT_ACTION(player, current_gameboard, allowable_moves, code, "make_out_of_turn_move")

def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    """
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param allowable_moves: A set of functions, each of which is defined in action_choices (imported in this file), and that
    will always be a subset of the action choices for post-die-roll in the game schema. Your returned action choice must be from
    allowable_moves; we will check for this when you return.
    :param code: See the preamble of this file for an explanation of this code
    :return: A 2-element tuple, the first of which is the action you want to take, and the second is a dictionary of
    parameters that will be passed into the function representing that action when it is executed.
    The dictionary must exactly contain the keys and expected value types expected by that action in
    action_choices
        """
    # for p in current_gameboard['players']:
    #     if 'phase_game' not in p.agent._agent_memory:
    #         p.agent._agent_memory['phase_game'] = 2
    #         p.agent._agent_memory['count_unsuccessful_tries'] = 0
    #
    # if player.agent._agent_memory['phase_game'] != 2:
    #     player.agent._agent_memory['phase_game'] = 2
    #     for p in current_gameboard['players']:
    #         if p.status != 'lost':
    #             p.agent._agent_memory['count_unsuccessful_tries'] = 0
    #
    # if code == flag_config_dict['failure_code']:
    #     player.agent._agent_memory['count_unsuccessful_tries'] += 1
    #     if player.agent.action is not None:
    #         player.agent.reward = torch.Tensor([-5]).to(player.agent.device)
    #         player.agent.next_state = player.agent.state.to(player.agent.device)
    #         rl_agent_helper.push_to_memory(player.agent)
    #         player.agent.failed_actions += 1
    #     logger.debug(player.player_name + ' has executed an unsuccessful postroll action, incrementing unsuccessful_tries ' +
    #                                       'counter to ' + str(player.agent._agent_memory['count_unsuccessful_tries']))

    if player.agent.action is not None:
        player.agent.successful_actions += 1
        player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
        player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
        rl_agent_helper.push_to_memory(player.agent)

    # if player.agent._agent_memory['count_unsuccessful_tries'] >= UNSUCCESSFUL_LIMIT:
    #     logger.debug(player.player_name + ' has reached postroll unsuccessful action limits.')
    #     if "concluded_actions" in allowable_moves:
    #         # player.agent._agent_memory['previous_action'] = action_choices.concluded_actions
    #         logger.debug(player.player_name+ ': I am concluding actions since I have crossed unsuccessful limits.')
    #         return ("concluded_actions", dict())
    #     else:
    #         logger.error("Exception")
    #         raise Exception
    return OUTPUT_ACTION(player, current_gameboard, allowable_moves, code, "make_post_roll_move")

def make_buy_property_decision(player, current_gameboard, asset):
    """
    The agent decides to buy the property if:
    (i) it can 'afford' it. Our definition of afford is that we must have at least go_increment cash balance after
    the purchase.
    (ii) we can obtain a full color set through the purchase, and still have positive cash balance afterwards (though
    it may be less than go_increment).

    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :return: A Boolean. If True, then you decided to purchase asset from the bank, otherwise False. We allow you to
    purchase the asset even if you don't have enough cash; however, if you do you will end up with a negative
    cash balance and will have to handle that if you don't want to lose the game at the end of your move (see notes
    in handle_negative_cash_balance)
    """
    if player.agent.action is not None:
        player.agent.successful_actions += 1
        player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
        player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
        rl_agent_helper.push_to_memory(player.agent)

    decision = False
    if player.current_cash - asset.price >= current_gameboard['go_increment']:  # case 1: can we afford it?
        logger.debug(player.player_name+ ': I will attempt to buy '+ asset.name+ ' from the bank.')
        decision = True
    elif asset.price <= player.current_cash and \
            agent_helper_functions_v2.will_property_complete_set(player,asset,current_gameboard):
        logger.debug(player.player_name+ ': I will attempt to buy '+ asset.name+ ' from the bank.')
        decision = True

    return decision


def make_bid(player, current_gameboard, asset, current_bid):
    """
    Decide the amount you wish to bid for asset in auction, given the current_bid that is currently going. If you don't
    return a bid that is strictly higher than current_bid you will be removed from the auction and won't be able to
    bid anymore. Note that it is not necessary that you are actually on the location on the board representing asset, since
    you will be invited to the auction automatically once a player who lands on a bank-owned asset rejects buying that asset
    (this could be you or anyone else).
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :param asset: An purchaseable instance of Location (i.e. real estate, utility or railroad)
    :param current_bid: The current bid that is going in the auction. If you don't bid higher than this amount, the bank
    will remove you from the auction proceedings. You could also always return 0 to voluntarily exit the auction.
    :return: An integer that indicates what you wish to bid for asset
    """
    if player.agent.action is not None:
        player.agent.successful_actions += 1
        player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
        player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
        rl_agent_helper.push_to_memory(player.agent)

    if current_bid < asset.price:
        new_bid = current_bid + (asset.price-current_bid)/2
        if new_bid < player.current_cash:
            return new_bid
        else:   # We are aware that this can be simplified with a simple return 0 statement at the end. However in the final baseline agent
                # the return 0's would be replaced with more sophisticated rules. Think of them as placeholders.
            return 0 # this will lead to a rejection of the bid downstream automatically
    elif current_bid < player.current_cash and agent_helper_functions_v2.will_property_complete_set(player,asset,
                                                                                                  current_gameboard):
            # We are prepared to bid more than the price of the asset only if it doesn't result in insolvency, and
                # if we can get a monopoly this way
        return current_bid+(player.current_cash-current_bid)/4
    else:
        return 0 # no reason to bid


def handle_negative_cash_balance(player, current_gameboard):
    """
    You have a negative cash balance at the end of your move (i.e. your post-roll phase is over) and you must handle
    this issue before we move to the next player's pre-roll. If you do not succeed in restoring your cash balance to
    0 or positive, bankruptcy proceeds will begin and you will lost the game.
    The background agent tries a number of things to get itself out of a financial hole. First, it checks whether
    mortgaging alone can save it. If not, then it begins selling unimproved properties in ascending order of price, the idea being
    that it might as well get rid of cheap properties. This may not be the most optimal move but it is reasonable.
    If it ends up selling all unimproved properties and is still insolvent, it starts selling improvements, followed
    by a sale of the (now) unimproved properties.
    :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
    instantiated with the functions specified by this decision agent).
    :param current_gameboard: A dict. The global data structure representing the current game board.
    :return: -1 (failure code) if you do not try to address your negative cash balance, or 1 if you tried and believed you succeeded.
    Note that even if you do return 1 (successful move action), we will check to see whether you have non-negative cash balance. The rule of thumb
    is to return 1 (successful move action) as long as you 'try', or -1 if you don't try (in which case you will be declared bankrupt and lose the game)
    """
    if player.agent.action is not None:
        player.agent.successful_actions += 1
        player.agent.next_state = STATE_FUNCTION(current_gameboard, player).to(player.agent.device)
        player.agent.reward = REWARD_FUNCTION(current_gameboard, player).to(player.agent.device)
        rl_agent_helper.push_to_memory(player.agent)

    if player.current_cash >= 0:   # prelim check to see if player has negative cash balance
        return (None, flag_config_dict['successful_action'])

    #player should evaluate all the possible options that can save it from bankruptcy and take most promising actions

    mortgage_potentials = list()
    max_sum = 0
    sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)
    for a in sorted_player_assets_list:
        if a.is_mortgaged:
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        else:
            mortgage_potentials.append((a, a.mortgage))
            max_sum += a.mortgage

    if mortgage_potentials and max_sum+player.current_cash >= 0: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action']) # we're done

            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player.player_name+ ': I am attempting to mortgage property '+ params['asset'])
            player.agent._agent_memory['previous_action'] = "mortgage_property"
            return ("mortgage_property", params)


    # if we got here, it means we're still in trouble. Next move is to sell unimproved properties. We don't check if
    # the total will cover our debts, since we're desperate at this point.

    # following sale potentials doesnot include properties from monopolized color groups
    sale_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)
    for a in sorted_player_assets_list:
        if a.color in player.full_color_sets_possessed:
            continue
        elif a.is_mortgaged:
            if (a.price*current_gameboard['bank'].property_sell_percentage)-Bank.calculate_mortgage_owed(a, current_gameboard) > 0:
                # default case, this will never be > 0 unless novelty is introduced
                sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-Bank.calculate_mortgage_owed(a, current_gameboard)))
            else:
                continue        # no point selling a mortgaged property if you dont anything out of it
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        else:
            sale_potentials.append((a, a.price*current_gameboard['bank'].property_sell_percentage))

    if sale_potentials: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action']) # we're done

            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player.player_name + ': I am attempting to sell property '+ p[0].name + ' to the bank')
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    # if selling properties from non monopolized color groups doesnot relieve the player from debt, then only we start thinking about giving up monopolized groups.
    # If we come across a unimproved property which belongs to a monopoly, we still have to loop through the other properties from the same color group and
    # sell the houses and hotels first because we cannot sell this property when the color group has improved properties
    # We first check if selling houses and hotels one by one on the other improved properties of the same color group relieves the player of his debt. If it does
    # then we return without selling the current property else we sell the property and the player loses monopoly of that color group.
    max_sum = 0
    sale_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)
    for a in sorted_player_assets_list:
        if a.is_mortgaged:
            if a.price*current_gameboard['bank'].property_sell_percentage-Bank.calculate_mortgage_owed(a, current_gameboard) > 0:
                # default case, this will never be > 0 unless novelty is introduced
                sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-Bank.calculate_mortgage_owed(a, current_gameboard)))
            else:
                continue   # no point selling mortgaged property if you dont get anything out of it
        elif a.loc_class=='real_estate' and (a.num_houses > 0 or a.num_hotels > 0):
            continue
        else:
            sale_potentials.append((a, a.price*current_gameboard['bank'].property_sell_percentage))

    if sale_potentials:
        sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by sell value in ascending order
        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action']) # we're done

            sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)
            for prop in sorted_player_assets_list:
                if prop != p[0] and prop.color == p[0].color and p[0].color in player.full_color_sets_possessed:
                    if prop.num_hotels > 0:    # if current asset has no hotels, prop can only have max of 1 hotel (uniform improvement rule)
                        if player.current_cash >= 0:
                            return (None, flag_config_dict['successful_action'])
                        params = dict()
                        params['player'] = player.player_name
                        params['asset'] = prop.name
                        params['current_gameboard'] = "current_gameboard"
                        params['sell_house'] = False
                        params['sell_hotel'] = True
                        logger.debug(player.player_name+ ': I am attempting to sell hotel on '+ prop.name + ' to the bank')
                        player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                        return ("sell_house_hotel", params)

                    elif prop.num_houses > 0:     # if current asset has no houses, prop can only have max of 1 house (uniform improvement rule)
                        if player.current_cash >= 0:
                            return (None, flag_config_dict['successful_action'])
                        params = dict()
                        params['player'] = player.player_name
                        params['asset'] = prop.name
                        params['current_gameboard'] = "current_gameboard"
                        params['sell_house'] = True
                        params['sell_hotel'] = False
                        logger.debug(player.player_name+ ': I am attempting to sell house on '+ prop.name + ' to the bank')
                        player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                        return ("sell_house_hotel", params)
                    else:
                        continue

            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player.player_name + ': I am attempting to sell property '+ p[0].name + ' to the bank')
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    #we reach here if the player still hasnot cleared his debt. The above loop has now resulted in some more non monopolized properties.
    #Hence we have to go through the process of looping through these properties once again to decide on the potential properties that can be mortgaged or sold.

    mortgage_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)
    for a in sorted_player_assets_list:
        if a.is_mortgaged:
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        else:
            mortgage_potentials.append((a,a.mortgage))

    if mortgage_potentials and max_sum+player.current_cash >= 0: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action']) # we're done

            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player.player_name+ ': I am attempting to mortgage property '+ params['asset'])
            player.agent._agent_memory['previous_action'] = "mortgage_property"
            return ("mortgage_property", params)

    # following sale potentials loops through the properties that have become unmonopolized due to the above loops and
    # doesnot include properties from monopolized color groups
    sale_potentials = list()
    sorted_player_assets_list = _set_to_sorted_list_assets(player.assets)

    for a in sorted_player_assets_list:
        if a.color in player.full_color_sets_possessed:
            continue
        elif a.is_mortgaged:
            if (a.price*current_gameboard['bank'].property_sell_percentage)-Bank.calculate_mortgage_owed(a, current_gameboard) > 0:
                sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-Bank.calculate_mortgage_owed(a, current_gameboard)))
            else:
                continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        else:
            sale_potentials.append((a, a.price*current_gameboard['bank'].property_sell_percentage))

    if sale_potentials: # if the second condition is not met, no point in mortgaging
        sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
        for p in sorted_potentials:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action']) # we're done

            params = dict()
            params['player'] = player.player_name
            params['asset'] = p[0].name
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player.player_name + ': I am attempting to sell property '+ p[0].name + ' to the bank')
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

    count = 0
    # if we're STILL not done, then the only option is to start selling houses and hotels from the remaining improved monopolized properties, if we have 'em
    while (player.num_total_houses > 0 or player.num_total_hotels > 0) and count <3: # often times, a sale may not succeed due to uniformity requirements. We keep trying till everything is sold,
        # or cash balance turns non-negative.
        count += 1 # there is a slim chance that it is impossible to sell an improvement unless the player does something first (e.g., replace 4 houses with a hotel).
        # The count ensures we terminate at some point, regardless.
        sorted_assets_list = _set_to_sorted_list_assets(player.assets)

        for a in sorted_assets_list:
            if a.loc_class == 'real_estate' and a.num_houses > 0:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done
                flag = True
                for same_colored_asset in current_gameboard['color_assets'][a.color]:
                    if same_colored_asset == a:
                        continue
                    if same_colored_asset.num_houses > a.num_houses or a.num_hotels == 1:
                        flag = False
                        break
                if flag:
                    params = dict()
                    params['player'] = player.player_name
                    params['asset'] = a.name
                    params['current_gameboard'] = "current_gameboard"
                    params['sell_house'] = True
                    params['sell_hotel'] = False
                    logger.debug(player.player_name+ ': I am attempting to sell house on '+ a.name + ' to the bank')
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)

            elif a.loc_class == 'real_estate' and a.num_hotels > 0:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done
                flag = True
                for same_colored_asset in current_gameboard['color_assets'][a.color]:
                    if same_colored_asset == a:
                        continue
                    if a.num_hotels == 1 and not (same_colored_asset.num_hotels == 1 or (same_colored_asset.num_hotels == 0 and
                                            same_colored_asset.num_houses == 0)) : # if there are no hotels on other properties,
                        # there must not be houses either, otherwise  the uniform improvement rule gets broken. The not on the
                        # outside enforces this rule.
                        flag = False
                        break
                    elif a.num_hotels < same_colored_asset.num_hotels:    # need to follow uniform improvement rule
                        flag = False
                        break
                if flag:
                    params = dict()
                    params['player'] = player.player_name
                    params['asset'] = a.name
                    params['current_gameboard'] = "current_gameboard"
                    params['sell_house'] = False
                    params['sell_hotel'] = True
                    logger.debug(player.player_name+ ': I am attempting to sell house on '+ a.name + ' to the bank')
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)

    # final straw
    final_sale_assets = player.assets.copy()
    sorted_player_assets_list = _set_to_sorted_list_assets(final_sale_assets)
    for a in sorted_player_assets_list:
        if player.current_cash >= 0:
            return (None, flag_config_dict['successful_action'])  # we're done
        if a.is_mortgaged:
            continue
        elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
            continue
        params = dict()
        params['player'] = player.player_name
        params['asset'] = a.name
        params['current_gameboard'] = "current_gameboard"
        logger.debug(player.player_name + ': I am attempting to sell property '+ a.name + ' to the bank')
        player.agent._agent_memory['previous_action'] = "sell_property"
        return ("sell_property", params)

    return (None, flag_config_dict['successful_action']) # if we didn't succeed in establishing solvency, it will get caught by the simulator. Since we tried, we return 1.


def _set_to_sorted_list_mortgaged_assets(player_mortgaged_assets):
    player_m_assets_list = list()
    player_m_assets_dict = dict()
    for item in player_mortgaged_assets:
        player_m_assets_dict[item.name] = item
    for sorted_key in sorted(player_m_assets_dict):
        player_m_assets_list.append(player_m_assets_dict[sorted_key])
    return player_m_assets_list


def _set_to_sorted_list_assets(player_assets):
    player_assets_list = list()
    player_assets_dict = dict()
    for item in player_assets:
        player_assets_dict[item.name] = item
    for sorted_key in sorted(player_assets_dict):
        player_assets_list.append(player_assets_dict[sorted_key])
    return player_assets_list

def _build_decision_agent_methods_dict():
    """
    This function builds the decision agent methods dictionary.
    :return: The decision agent dict. Keys should be exactly as stated in this example, but the functions can be anything
    as long as you use/expect the exact function signatures we have indicated in this document.
    """
    ans = dict()
    ans['handle_negative_cash_balance'] = handle_negative_cash_balance
    ans['make_pre_roll_move'] = make_pre_roll_move
    ans['make_out_of_turn_move'] = make_out_of_turn_move
    ans['make_post_roll_move'] = make_post_roll_move
    ans['make_buy_property_decision'] = make_buy_property_decision
    ans['make_bid'] = make_bid
    ans['type'] = "decision_agent_methods"
    return ans


decision_agent_methods = _build_decision_agent_methods_dict()  # this is the main data structure that is needed by
# gameplay
