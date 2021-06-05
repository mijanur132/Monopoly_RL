import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple
from itertools import count

import random
import math
from monopoly_simulator import location, bank, action_choices, agent_helper_functions_v2, background_agent_v4_base, \
    background_agent_v5_base
import matplotlib.pyplot as plt
import numpy as np


class DQN(nn.Module):
    """ DQN Class"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1024)
        self.out = nn.Linear(in_features=1024, out_features=out_dim)

    def forward(self, t):
        # t = t.flatten()
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)


# REPLAY MEMORY
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


# EPSILON GREEDY
class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.rate = self.start

    def get_exploration_rate(self, current_step):
        # return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
        # print(self.rate)
        return self.rate


# EXTRACT TENSORS
def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


# Get current and next q-values
class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(1, actions)

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

    @staticmethod
    def get_next_v2(policy_net, target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)

        for idx, next_state in enumerate(next_states):
            Q = target_net(next_state).detach()
            amax = policy_net(next_state).argmax().detach()
            V = Q[amax]
            V += 0.1 * np.log(np.exp((Q - Q.max()) / 0.1).sum())
            values[idx] = V
        return values
        # values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        # return values


def push_to_memory(agent):
    """ Push Experience (state, action, reward, next_state) onto memory"""
    agent.episodic_rewards += agent.reward
    agent.memory.push(Experience(agent.state, agent.action,
                                 agent.next_state, agent.reward))
    agent.current_state = agent.next_state
    agent.next_state = None


# def plot_r_val(mean_r_val_list, num):
#     plt.figure(figsize=(18,5))
#     plt.plot(range(len(mean_r_val_list)), mean_r_val_list)
#     plt.subplot(111)
#     plt.title('Avg Rewards Values')
#     plt.ylabel('Avg Reward-Values')
#     plt.xlabel('Episodes')
#     # plt.show()
#     plt.savefig("R_"+str(num))

def plot_val(mean_q_val_list, avg_wins_list, mean_r_val_list, num):
    # print(mean_q_val_list)
    plt.figure(figsize=(18, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.subplot(131)
    plt.plot(range(len(mean_q_val_list)), mean_q_val_list)
    plt.title('Avg Q Values')
    plt.ylabel('Avg Q-Values')
    plt.xlabel('Episodes')
    plt.subplot(132)
    plt.title("RL Wins per 1000 games")
    plt.plot(list(avg_wins_list.keys()), list(avg_wins_list.values()))
    plt.ylabel('Avg Wins per 1000 games')
    plt.xlabel('Episodes')
    plt.subplot(133)
    plt.plot(range(len(mean_r_val_list)), mean_r_val_list)
    plt.title('Avg Reward Values')
    plt.ylabel('Avg Reward-Values')
    plt.xlabel('Episodes')
    plt.savefig("Avg_wins_Q_R_v5_"+str(num))

def get_state_vector(game_elements, player):
    """ First state vector - 277 dimensions - original"""
    v = []
    for player in game_elements['players']:
        # becomes none when won or lost the game
        if player.current_position is not None:
            v.append(player.current_position)
        else:
            v.append(-1)

        v.append(player.current_cash)

        # player.status: A string. One of 'waiting_for_move', 'current_move', 'won' or 'lost'
        if player.status == "waiting_for_move":
            v.append(1)
        else:
            v.append(0)

        if player.status == "current_move":
            v.append(1)
        else:
            v.append(0)

        if player.status == "won":
            v.append(1)
        else:
            v.append(0)

        if player.status == "lost":
            v.append(1)
        else:
            v.append(0)

        v.append(player.num_railroads_possessed)

        if player.currently_in_jail:
            v.append(1)
        else:
            v.append(0)

        v.append(player.num_utilities_possessed)

        if player.has_get_out_of_jail_community_chest_card:
            v.append(1)
        else:
            v.append(0)

        if player.has_get_out_of_jail_chance_card:
            v.append(1)
        else:
            v.append(0)

    # end of player based information

    v.append(game_elements['jail_position'])

    # railroad positions. this value depends on game_schema.
    # at the time of writing, there were 4 integer position values.

    for pos in game_elements['railroad_positions']:
        v.append(pos)

    # total number of railroads
    v.append(len(game_elements['railroad_positions']))

    # utility positions. this value depends on game_schema.
    # at the time of writing, there were 2 integer position values.

    for pos in game_elements['utility_positions']:
        v.append(pos)

    # total number of utilities
    v.append(len(game_elements['utility_positions']))

    # information about all objects
    for obj_name in game_elements['location_objects']:
        loc = game_elements['location_objects'][obj_name]

        # some location objects do not have any useful information. thus, we skip them
        if isinstance(loc, location.RealEstateLocation):
            # owner could be either Bank, or a player
            if isinstance(loc.owned_by, bank.Bank):
                v.append(1)
            else:
                v.append(0)

            # if it is owned by any of the players
            for player in game_elements['players']:
                if loc.owned_by == player:
                    v.append(1)
                else:
                    v.append(0)
            v.append(loc.price)
            v.append(loc.mortgage)
            if loc.is_mortgaged:
                v.append(1)
            else:
                v.append(0)
            pass
        if isinstance(loc, location.RailroadLocation):
            # owner could be either Bank, or a player
            if isinstance(loc.owned_by, bank.Bank):
                v.append(1)
            else:
                v.append(0)

            # if it is owned by any of the players
            for player in game_elements['players']:
                if loc.owned_by == player:
                    v.append(1)
                else:
                    v.append(0)

            v.append(loc.price)
            v.append(loc.mortgage)

            if loc.is_mortgaged:
                v.append(1)
            else:
                v.append(0)
            pass
        if isinstance(loc, location.UtilityLocation):
            # owner could be either Bank, or a player
            if isinstance(loc.owned_by, bank.Bank):
                v.append(1)
            else:
                v.append(0)

            # if it is owned by any of the players
            for player in game_elements['players']:
                if loc.owned_by == player:
                    v.append(1)
                else:
                    v.append(0)
            v.append(loc.price)
            v.append(loc.mortgage)

            if loc.is_mortgaged:
                v.append(1)
            else:
                v.append(0)
            pass
    return torch.as_tensor(v).unsqueeze(-2)


def get_state_vector2(game_elements, current_player):
    """Second state vector - dimensions - 23 - based on Bailis et. al."""
    all_asset_groups = get_all_asset_groups(game_elements)

    all_assets = get_all_assets(game_elements)

    active_players = get_active_players(game_elements)

    # -----

    # AREA OBJECT

    # 2 columns, 10 rows. access by [column][row]
    # set all percentages to zero.
    # column[0] for current_player
    # column[1] for other (opponent) players
    # each row represents a different asset group and how much of it is owned by current player and other players.
    area_object = [[0 for i in range(10)] for j in range(2)]

    # for each row (asset group)
    for asset_group_index, asset_group in enumerate(all_asset_groups):

        house_counter_of_current_player = 0
        house_counter_of_other_players = 0

        total_number_of_houses = 0

        number_of_properties_of_current_player = 0
        number_of_properties_of_other_players = 0

        total_number_of_properties = 0

        for asset in all_assets:
            if get_asset_group(asset) is not asset_group:
                continue

            total_number_of_properties += 1

            is_owned_by_current_player = (asset.owned_by is current_player)
            is_owned_by_active_players = (asset.owned_by in active_players)
            is_realestate = isinstance(asset, location.RealEstateLocation)

            # maximum 4 houses on a realestate. 1 hotel = 4 houses. max = 4 + 4 = 8.
            if is_realestate:
                total_number_of_houses += 8

                if is_owned_by_current_player:
                    house_counter_of_current_player += (asset.num_houses + 4 * asset.num_hotels)

                elif is_owned_by_active_players:
                    house_counter_of_other_players += (asset.num_houses + 4 * asset.num_hotels)

            # count number of owned properties
            if is_owned_by_current_player:
                number_of_properties_of_current_player += 1

            elif is_owned_by_active_players:
                number_of_properties_of_other_players += 1

        # -----

        area_of_current_player = 0
        area_of_other_players = 0

        # property based percentage
        if total_number_of_properties > 0:
            area_of_current_player = (number_of_properties_of_current_player / total_number_of_properties)
            area_of_other_players = (number_of_properties_of_other_players / total_number_of_properties)

        # house/hotel based percentage
        if total_number_of_houses > 0:
            area_of_current_player = (area_of_current_player + (
                        house_counter_of_current_player / total_number_of_houses)) / 2.0
            area_of_other_players = (area_of_other_players + (
                        house_counter_of_other_players / total_number_of_houses)) / 2.0

        # store the owned percentages
        area_object[0][asset_group_index] = area_of_current_player
        area_object[1][asset_group_index] = area_of_other_players

    # ----- -----

    # PREPARING POSITION VARIABLE

    location_object_at_player_position = game_elements['location_sequence'][current_player.current_position]
    asset_group_at_player_position = location_object_at_player_position.color

    if isinstance(location_object_at_player_position, location.RailroadLocation):
        asset_group_at_player_position = "Railroad"

    elif isinstance(location_object_at_player_position, location.UtilityLocation):
        asset_group_at_player_position = "Utility"

    # if asset_group_at_player_position is None:
    #    print("Asset group is None. Type: ", type(location_object_at_player_position))

    # --

    # find asset group position the player is on
    # .index might throw if asset group is not found.
    asset_group_index_in_all = 0

    if asset_group_at_player_position is not None:
        asset_group_index_in_all = all_asset_groups.index(asset_group_at_player_position)

    position_variable = asset_group_index_in_all / (len(all_asset_groups) - 1)

    # ----- -----

    # FINANCE VECTOR

    number_of_properties_current_player_has = len(get_all_assets_of_player(current_player))

    number_of_properties_other_players_has = 0

    for other_player in active_players:
        if other_player is current_player:
            continue

        number_of_properties_other_players_has += len(get_all_assets_of_player(other_player))

    part1 = (number_of_properties_current_player_has * 100 + 1)
    part2 = ((number_of_properties_current_player_has + number_of_properties_other_players_has) * 100 + 1)

    finance_vector_0 = (part1 / part2)

    # --

    finance_vector_1 = ((current_player.current_cash/1500) / (1 + abs((current_player.current_cash/1500))))

    # ----- -----

    # RESULTING VECTOR

    state_vector = []

    for i in range(2):
        for j in range(10):
            state_vector.append(area_object[i][j])

    state_vector.append(position_variable)

    state_vector.append(finance_vector_0)
    state_vector.append(finance_vector_1)

    return torch.as_tensor(state_vector).unsqueeze(-2)


def select_action(agent, state, policy_net, allowed_actions):
    """
    First version of select action, obsolete.
    :param agent:
    :param state:
    :param policy_net:
    :param allowed_actions: a vector that gives a list of allowed actions
    :return: allowed_action integer
    """
    rate = agent.strategy.get_exploration_rate(agent.current_step)
    agent.current_step += 1

    # Exploration or Exploitation
    # Returns one action - should be a valid move in the current state
    if rate > random.random():
        # For random, we can pass the allowable_moves vector and choose from it randomly
        action = random.choices(allowed_actions)

        # Originally
        # action = random.randrange(self.num_actions)
        # print('random action shape ', torch.tensor([action]).shape)
        return torch.tensor([action]).to(agent.device)  # explore
    else:
        with torch.no_grad():
            # ONLY RETURN VALID MOVES:
            # For the network, we could set the probabilities of all illegal moves to zero and
            # re-normalise the output vector before choosing the move.
            # OR we could just get the probabilities of allowed moves and choose the highest.
            # print('outcome of policy shape', policy_net(state).shape)
            # print(policy_net(state).argmax(dim=0))
            # print('net action shape ', policy_net(state).argmax(dim=1).shape)
            # Filter the output to only have allowed_actions
            output_vector = policy_net(state).gather(1, torch.tensor(allowed_actions).unsqueeze(-2))
            # return policy_net(state).max(1)[1].view(1, 1).to(agent.device)  # exploit
            return torch.tensor(allowed_actions[output_vector.max(1)[1]]).view(1, 1).to(agent.device)


def select_action_v2(agent, state, policy_net):
    """
    Second version of select action, episodic_step count kept.
    Only the one action is returned.
    :param agent:
    :param state:
    :param policy_net:
    :return:
    """
    rate = agent.strategy.get_exploration_rate(agent.current_step)
    agent.current_step += 1
    agent.episodic_step += 1
    # Exploration or Exploitation
    # Returns one action - should be a valid move in the current state
    if rate > random.random():
        # For random, we can pass the allowable_moves vector and choose from it randomly
        action = random.choice(range(agent.output_dim))
        # Originally
        # action = random.randrange(self.num_actions)
        # print('random action shape ', torch.tensor([action]).shape)
        return torch.tensor([[action]]).to(agent.device)  # explore
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1).to(agent.device)  # exploit
            # return torch.tensor(allowed_actions[output_vector.max(1)[1]]).view(1, 1).to(agent.device)

#
# def select_action_v3(agent, policy_net):
#     """
#     Third version of select action - same as v2 - but returns a list of ordered actions instead of one action.
#     :param agent:
#     :param state:
#     :param policy_net:
#     :return:
#     """
#     rate = agent.strategy.get_exploration_rate(agent.current_step)
#     agent.current_step += 1
#     agent.episodic_step += 1
#     # Exploration or Exploitation
#     if rate > random.random():
#         actions = random.sample(range(agent.output_dim), k=agent.output_dim)
#         return actions
#     else:
#         with torch.no_grad():
#             return torch.sort(policy_net(agent.state), descending=True)[1].data.detach().tolist()[0]

def select_action_v3(agent, policy_net, move, player, current_gameboard, allowable_moves, code):
    """
    Third version of select action - same as v2 - but returns a list of ordered actions instead of one action.
    :param agent:
    :param state:
    :param policy_net:
    :return:
    """
    rate = agent.strategy.get_exploration_rate(agent.current_step)


    agent.current_step += 1
    agent.episodic_step += 1
    # Exploration or Exploitation
   # if rate > random.random():
    if 1!=1:
         #print(random.random())
        # actions = random.sample(range(agent.output_dim), k=agent.output_dim)
        actions = list()
        act = background_agent_v5_base.decision_agent_methods[move](player, current_gameboard, allowable_moves, code)
        actions.append(act)
        # print("V3 Actions: ")
        return actions
    else:
        # print("Active Actions: ")
        with torch.no_grad():
            actions= torch.sort(policy_net(agent.state), descending=True)[1].data.detach().tolist()[0]
            #print("rl_agent_helper, 569, actions:", actions)
            return actions

def select_action_v3_sac(agent, move, player, current_gameboard, allowable_moves, code):
    """
    Third version of select action - same as v3 but for sac
    :param agent:
    :param state:
    :return:
    """
    rate = agent.strategy.get_exploration_rate(agent.current_step)
    agent.current_step += 1
    agent.episodic_step += 1
    # Exploration or Exploitation
    if rate > random.random():
        #print(random.random())
        # actions = random.sample(range(agent.output_dim), k=agent.output_dim)
        actions = list()
        act = background_agent_v5_base.decision_agent_methods[move](player, current_gameboard, allowable_moves, code)
        actions.append(act)
        # print("V3 Actions: ")
        return actions
    else:
        # print("Active Actions: ")
        with torch.no_grad():

           actions= torch.sort(policy_net(agent.state), descending=True)[1].data.detach().tolist()[0]

           return actions


def select_action_v4(agent, policy_net, allowable_moves):
    """
    Third version of select action - same as v2 - but returns a list of ordered actions instead of one action.
    :param agent:
    :param state:
    :param policy_net:
    :return:
    """
    allowed_actions = set()
    for i in allowable_moves:
        if i != 'make_sell_property_offer':
            if i == 'sell_house_hotel':
                allowed_actions.update(agent.action_map_r['sell_property'])
            else:
                allowed_actions.update(agent.action_map_r[i])
    rate = agent.strategy.get_exploration_rate(agent.current_step)
    agent.current_step += 1
    agent.episodic_step += 1
    # Exploration or Exploitation
    if rate > random.random():
        actions = random.sample(allowed_actions, k=len(allowed_actions))
        print("rl_agent_helper, 593, selected_actions:", actions)
        return actions
    else:
        with torch.no_grad():
            actions= torch.sort(policy_net(agent.state), descending=True)[1].data.detach().tolist()[0]
            print("rl_agent_helper, 599, selected_actions:", actions)
            return actions

def get_action_vector5(game_elements): #current usage
    # -----

    simplified_actions_with_color = [
        "free_mortgage",
        # "make_sell_property_offer",
        "sell_property",
        "buy_property",
        "mortgage_property",
        "improve_property",
        "make_trade_offer"
    ]

    simplified_actions_without_color = [
        "concluded_actions",
        "use_get_out_of_jail_card",
        "skip_turn",
        "pay_jail_fine",
        "accept_trade_offer",
        "accept_sell_property_offer"
    ]

    # -----

    colors = []

    for color_asset in game_elements['color_assets']:
        colors.append(color_asset)

    colors.append("Railroad")
    colors.append("Utility")

    # -----

    action_vector = {}
    action_key = 0

    for action in simplified_actions_with_color:
        for color in colors:
            action_vector[action_key] = {"loc_group": color, "action_type": action}
            action_key += 1

    for action in simplified_actions_without_color:
        action_vector[action_key] = {"loc_group": None, "action_type": action}
        action_key += 1

    # -----

    return action_vector


# using owned, mortgaged assets
def get_action_vector_mapping__free_mortgage(game_elements, player, asset_group):
    mortgaged_assets = []

    # collect all mortgaged assets
    for asset in get_all_assets_of_player(player):
        if asset.is_mortgaged:
            mortgaged_assets.append(asset)

    # order them from lowest mortgage to highest
    mortgaged_assets = sorted(mortgaged_assets, key=lambda asset: asset.mortgage, reverse=False)

    # -----

    # if no asset is available, skip this turn
    if len(mortgaged_assets) <= 0:
        return (None, None, True)

    else:
        return ("free_mortgage",
                {"player": player, "asset": mortgaged_assets[0], "current_gameboard": game_elements}, False)


# ----------------------------------


# using owned, not mortgaged assets
# check agent_helper_functions.identify_sale_opportunity_to_player
def get_action_vector_mapping__make_sell_property_offer(game_elements, player, asset_group):
    offers = []

    for asset in get_all_assets_of_player(player):
        # skip mortgaged assets
        if asset.is_mortgaged:
            continue

        # if asset group doesn't match, skip it
        if get_asset_group(asset) != asset_group:
            continue

        price = asset.price * 1.5  # 50% markup on market price.

        for other_player in get_active_players(game_elements):
            # if same player, skip it
            if other_player == player:
                continue

            if price < other_player.current_cash / 2:
                price = other_player.current_cash / 2  # how far would you go for a monopoly?

            elif price > other_player.current_cash:
                # no point offering this to the player; they don't have money.
                continue

            # create the offer
            offers.append({"player": other_player, "asset": asset, "price": price})

    # -----

    # if no offer is available, skip this turn
    if len(offers) <= 0:
        return (None, None, True)
    else:
        # highest price to lowest price
        sorted_offers = sorted(offers, key=lambda offer: offer["price"], reverse=True)

        return ("make_sell_property_offer",
                {"from_player": player, "asset": sorted_offers[0]["asset"], "to_player": sorted_offers[0]["player"],
                 "price": sorted_offers[0]["price"]}, False)


# ----------------------------------


# using owned, NOT mortgaged assets
def get_action_vector_mapping__sell_property(game_elements, player, asset_group):
    realestate_assets = []
    other_assets = []

    number_of_owned_railroads = len(get_assets_of_player_by_group(player, "Railroad"))

    for asset in get_all_assets_of_player(player):
        is_realestate = isinstance(asset, location.RealEstateLocation)
        is_railroad = isinstance(asset, location.RailroadLocation)

        # skip mortgaged assets
        if asset.is_mortgaged:
            continue

        # if asset group doesn't match, skip it
        if get_asset_group(asset) != asset_group:
            continue

        # Strategy: never sell fourth railroad
        if is_railroad and (number_of_owned_railroads >= 4):
            continue

        # check type
        if is_realestate:
            realestate_assets.append(asset)
        else:
            other_assets.append(asset)

    # -----

    # check whether there is any hotel in any of real estates
    hotel_exists = False

    for asset in realestate_assets:
        if asset.num_hotels > 0:
            hotel_exists = True
            break

            # -----

    # if there are houses, get maximum number of houses
    max_number_of_houses = 0

    for asset in realestate_assets:
        if asset.num_houses > max_number_of_houses:
            max_number_of_houses = asset.num_houses

    # -----

    assets_to_sell = []

    # start with real estates
    for asset in realestate_assets:
        # if there is any hotel, ...
        if hotel_exists:
            # before hotels, we cannot sell houses
            if asset.num_hotels <= 0:
                continue

            # sell the hotel
            assets_to_sell.append(
                {"action": "sell_house_hotel", "asset": asset, "sellHouse": False, "sellHotel": True})

        # if there is no hotel, but there are houses
        elif (max_number_of_houses > 0):
            # houses must be sold evenly.
            if asset.num_houses != max_number_of_houses:
                continue

            # sell the house
            assets_to_sell.append(
                {"action": "sell_house_hotel", "asset": asset, "sellHouse": True, "sellHotel": False})

        # if there neigther any hotel or house, sell the property
        else:
            assets_to_sell.append(
                {"action": "sell_property", "asset": asset, "sellHouse": None, "sellHotel": None})

    # sell non real estates
    for asset in other_assets:
        assets_to_sell.append(
            {"action": "sell_property", "asset": asset, "sellHouse": None, "sellHotel": None})

    # -----

    # if no asset is available, skip this turn
    if len(assets_to_sell) <= 0:
        return (None, None, True)

    else:
        if assets_to_sell[0]["action"] == "sell_property":
            # print ("aaaaaaa",assets_to_sell[0]["asset"])
            return assets_to_sell[0]["action"], {"player": player, "asset": assets_to_sell[0]["asset"],
                                                 "current_gameboard": game_elements}, False
        else:
            if assets_to_sell[0]["sellHouse"]:
                # print("bbbbbb", assets_to_sell[0]["asset"])
                return (assets_to_sell[0]["action"],
                        {"player": player, "asset": assets_to_sell[0]["asset"], "current_gameboard": game_elements,
                         "sell_house": True, "sell_hotel": False}, False)
            else:
                # print("cccccc", assets_to_sell[0]["asset"])
                return (assets_to_sell[0]["action"],
                        {"player": player, "asset": assets_to_sell[0]["asset"], "current_gameboard": game_elements,
                         "sell_house": False, "sell_hotel": True}, False)
    # ----------------------------------


# using NOT owned, NOT mortgaged assets
# properties are only bought from Bank, not other players. (written in action_choices.py)
def get_action_vector_mapping__buy_property(game_elements, player, asset_group):
    possible_assets = []

    for asset in get_all_assets_not_of_player(game_elements, player):
        # skip mortgaged assets
        if asset.is_mortgaged:
            continue

        # property must be owned by bank
        if not isinstance(asset.owned_by, bank.Bank):
            continue

        # if asset group doesn't match, skip it
        if get_asset_group(asset) != asset_group:
            continue

        # add this property to list
        possible_assets.append(asset)

    # if there is no property we can buy, skip this turn
    if len(possible_assets) <= 0:
        return (None, None, True)

    # -----

    chosen_asset = None

    active_players = get_active_players(game_elements)

    # -----

    # Strategy 1: Buy as many orange properties as possible. Players go into orange locations
    # after getting out of jail mostly.
    if (chosen_asset is None) and (asset_group == "Orange"):
        # choose any asset. all of them are in the same group already anyway.
        chosen_asset = possible_assets[0]

    # -----

    # Strategy 2: If a color group is not owned by any other player, buy that one
    if chosen_asset is None:
        strategy_failed = False

        for other_player in active_players:
            # make sure it is not the same player
            if other_player is player:
                continue

            # does this player own any asset within this group?
            if len(get_assets_of_player_by_group(other_player, asset_group)) > 0:
                strategy_failed = True
                break

        # if strategy didn't fail, choose any asset from list
        if not strategy_failed:
            chosen_asset = possible_assets[0]

    # -----

    # Strategy 3: If buying an asset in this asset group would make the player has 2 or more of the same
    # asset group property, buy.
    if chosen_asset is None:
        number_of_owned_properties_in_this_group = len(get_assets_of_player_by_group(player, asset_group))

        if number_of_owned_properties_in_this_group >= 1:
            chosen_asset = possible_assets[0]

    # -----

    # Strategy 4: If buying an asset in this asset group would prevent an opponent to reach 2 or more of the same
    # asset group property, buy.
    if chosen_asset is None:
        any_opponent_fits = False

        # if any opponent has more than 0 property in this asset group, buy.
        for other_player in active_players:
            # make sure it is not the same player
            if other_player is player:
                continue

            # does this player own any asset within this group?
            if len(get_assets_of_player_by_group(other_player, asset_group)) > 0:
                any_opponent_fits = True
                break

        # isn't this same as strategy 2?
        if any_opponent_fits:
            chosen_asset = possible_assets[0]

    # -----

    # if none of the strategies worked out, do not play
    if chosen_asset is None:
        return (None, None, True)
    else:
        return (
            "buy_property", {"player": player, "asset": chosen_asset, "current_gameboard": game_elements},
            False)


# ----------------------------------


# using NOT owned, NOT mortgaged assets
def get_action_vector_mapping__mortgage_property(game_elements, player, asset_group):
    # if the player owns two or more the the same group asset, do not play.
    # if at least one property in an asset group is mortgaged, you cannot build on any of the properties in that asset group.
    if len(get_assets_of_player_by_group(player, asset_group)) >= 2:
        return (None, None, True)

    # collect all not owned, not mortgaged assets
    not_mortgaged_assets = []

    for asset in get_all_assets_not_of_player(game_elements, player):
        # if asset is already mortgaged, skip it
        if asset.is_mortgaged:
            continue

        # possible asset
        not_mortgaged_assets.append(asset)

    # order them from highest mortgage to lowest
    not_mortgaged_assets = sorted(not_mortgaged_assets, key=lambda asset: asset.mortgage, reverse=True)

    # -----

    # if no asset is available, skip this turn
    if len(not_mortgaged_assets) <= 0:
        return (None, None, True)
    else:
        return ("mortgage_property",
                {"player": player, "asset": not_mortgaged_assets[0], "current_gameboard": game_elements}, False)


# ----------------------------------


# using owned, not mortgaged assets
def get_action_vector_mapping__improve_property(game_elements, player, asset_group):
    suitable_assets = []

    # asset must match the asset group, also it must be a RealEstate
    for asset in get_all_assets_of_player(player):
        # mortgaged assets cannot be improved
        if asset.is_mortgaged:
            continue

        # must be a Real Estate
        if not isinstance(asset, location.RealEstateLocation):
            continue

        # must match asset group
        if get_asset_group(asset) != asset_group:
            continue

        # only one hotel is allowed per property. if there is a hotel, we cannot improve it.
        if asset.num_hotels > 0:
            continue

        suitable_assets.append(asset)

    # -----

    # if there is no suitable asset, skip turn
    if len(suitable_assets) <= 0:
        return (None, None, True)

    # -----

    assets_to_improve = []

    # house must be distributed evenly for all properties.
    # we sort them from lowest number of houses to highest.
    suitable_assets = sorted(suitable_assets, key=lambda asset: asset.num_houses, reverse=False)

    # if minimum number of houses is 4, next step would be adding a hotel for all properties
    if suitable_assets[0].num_houses == 4:

        for asset in suitable_assets:
            assets_to_improve.append({"asset": asset, "addHouse": False, "addHotel": True})

    # if minimum number of houses hasn't reached 4 for all properties, we need to add house
    else:
        # if there are 3 properties, and there are 1, 1, and 2 houses on them,
        # we can add a new house to those first two properties.
        allowed_number_of_houses = suitable_assets[0].num_houses

        for asset in suitable_assets:
            if asset.num_houses == allowed_number_of_houses:
                assets_to_improve.append({"asset": asset, "addHouse": True, "addHotel": False})

    # if there is no suitable asset, skip turn
    if len(assets_to_improve) <= 0:
        return (None, None, True)
    else:
        return ("improve_property",
                {"player": player, "asset": assets_to_improve[0]["asset"], "current_gameboard": game_elements,
                 "add_house": assets_to_improve[0]["addHouse"], "add_hotel": assets_to_improve[0]["addHotel"]}, False)


# ----------------------------------


def calculate_weight__make_trade_offer(
        group_of_offered_asset,
        asset_offered,
        asset_wanted,
        total_number_of_assets_in_offered_group,
        owned_number_of_assets_in_offered_group,
        other_player_owned_number_of_assets_in_offered_group):
    group_of_wanted_asset = get_asset_group(asset_wanted)

    # if acceptance of this offer makes the player own all assets of same group, make the offer. this is the best offer.
    if (group_of_wanted_asset == group_of_offered_asset) and (
            (owned_number_of_assets_in_offered_group + 1) == total_number_of_assets_in_offered_group):
        return 8

    # if acceptance of this offer makes other player own all assets of same group, do not make the offer.
    if (group_of_offered_asset == group_of_wanted_asset) and (
            (other_player_owned_number_of_assets_in_offered_group + 1) == total_number_of_assets_in_offered_group):
        return 1

    # otherwise, start with RealEstate offerings
    if not ((group_of_offered_asset == "Railroad") or (group_of_offered_asset == "Utility")):
        if group_of_offered_asset == "Orange":
            return 7

        elif group_of_offered_asset == "SkyBlue":
            return 6

        elif group_of_offered_asset == "Orchid":
            return 5

        else:
            return 4

    # then offer RailRoads
    elif group_of_offered_asset == "Railroad":
        return 3

    # then offer Utilities
    elif group_of_offered_asset == "Utility":
        return 2

    # unknown
    return 0


# using owned assets
def get_action_vector_mapping__make_trade_offer(game_elements, player, offered_asset_group):
    # count assets of given asset group
    total_number_of_assets_in_this_group = 0

    for other_asset_group in get_all_asset_groups(game_elements):
        if other_asset_group == offered_asset_group:
            total_number_of_assets_in_this_group

    player_assets = get_assets_of_player_by_group(player, offered_asset_group)

    owned_number_of_assets_in_this_group = len(player_assets)

    chosen_weight = -1
    chosen_to_player = None
    chosen_asset_offered = None
    chosen_asset_wanted = None

    for asset_offered in player_assets:
        # offered asset's group must match
        if get_asset_group(asset_offered) != offered_asset_group:
            continue

        for other_player in get_active_players(game_elements):
            # if same player, skip it
            if other_player == player:
                continue

            other_player_same_group_assets = get_assets_of_player_by_group(other_player, offered_asset_group)
            other_player_owned_number_of_assets_in_this_group = len(other_player_same_group_assets)

            for asset_wanted in get_all_assets_of_player(other_player):
                # calculate worth of this offer
                weight = calculate_weight__make_trade_offer(offered_asset_group, asset_offered, asset_wanted,
                                                            total_number_of_assets_in_this_group,
                                                            owned_number_of_assets_in_this_group,
                                                            other_player_owned_number_of_assets_in_this_group)

                # if this is a better deal, then choose it
                if weight > chosen_weight:
                    chosen_weight = weight
                    chosen_to_player = other_player
                    chosen_asset_offered = asset_offered
                    chosen_asset_wanted = asset_wanted

    # -----

    # if no offer was possible, skip this turn
    if chosen_to_player is None:
        return (None, None, True)
    else:
        # cash offered for the respective requested property
        ask_price = chosen_asset_offered.price * 1.5  # 50% markup on market price.

        # cash received for the respective offered property
        bid_price = chosen_asset_wanted.price * 1.4

        param = agent_helper_functions_v2.curate_trade_offer(
            player,
            [[{"to_player": chosen_to_player, "asset": chosen_asset_offered, "price": ask_price}]],
            [[{"from_player": player, "asset": chosen_asset_wanted, "price": bid_price}]],
            game_elements,
            2
        )

        if param is None:
            return (None, None, True)
        else:
            return ("make_trade_offer",
                    {"from_player": param["from_player"], "offer": param["offer"], "to_player": param["to_player"]},
                    False)

    # SHOULD WE IGNORE OUR OWN WRITTEN "CALCULATE WEIGHT" FUNCTION, AND JUST USE "CURATE_TRADE_OFFER" WITH ALL POSSIBLE ASSETS?


# ----------------------------------


def get_action_vector_mapping(action_idx, player, allowable_moves, current_gameboard):
    if not is_player_active(player):
        return (None, None, True)

    # -----

    chosen_action_vector_item = get_action_vector5(current_gameboard)[action_idx]

    # -----

    simplified_actions_with_color = [
        "free_mortgage",
        "make_sell_property_offer",
        "sell_property",
        "buy_property",
        "mortgage_property",
        "improve_property",
        "make_trade_offer",
    ]

    simplified_actions_without_color = [
        "concluded_actions",
        "use_get_out_of_jail_card",
        "skip_turn",
        "pay_jail_fine",
        "accept_trade_offer",
        "accept_sell_property_offer"
    ]

    # -----

    # if chosen action is not allowed, skip turn

    allowedActions = allowable_moves

    actionAllowed = False

    for actionName in allowedActions:
        # actionName = act.__name__

        if chosen_action_vector_item["action_type"] == actionName:
            actionAllowed = True
            break

    if not actionAllowed:
        return (None, None, True)

    # -----

    # get the action name and chosen asset group

    action = chosen_action_vector_item["action_type"]
    asset_group = chosen_action_vector_item["loc_group"]

    # action does not need any asset
    if action == "concluded_actions":
        return ("concluded_actions", dict(), False)

    elif action == "use_get_out_of_jail_card":
        return (
            "use_get_out_of_jail_card", {"player": player, "current_gameboard": current_gameboard}, False)

    elif action == "skip_turn":
        return ("skip_turn", dict(), False)

    elif action == "pay_jail_fine":
        return ("pay_jail_fine", {"player": player, "current_gameboard": current_gameboard}, False)

    elif action == "accept_trade_offer":
        return ("accept_trade_offer", {"player": player, "current_gameboard": current_gameboard}, False)

    elif action == "accept_sell_property_offer":
        return (
            "accept_sell_property_offer", {"player": player, "current_gameboard": current_gameboard},
            False)

    # -----

    # using owned assets
    if action == "free_mortgage":
        return get_action_vector_mapping__free_mortgage(current_gameboard, player, asset_group)

    # using owned assets
    elif action == "make_sell_property_offer":
        return get_action_vector_mapping__make_sell_property_offer(current_gameboard, player, asset_group)

    # using owned assets
    elif action == "sell_property":
        return get_action_vector_mapping__sell_property(current_gameboard, player, asset_group)

    # using NOT owned assets
    elif action == "buy_property":
        return get_action_vector_mapping__buy_property(current_gameboard, player, asset_group)

    # using NOT owned assets
    elif action == "mortgage_property":
        return get_action_vector_mapping__mortgage_property(current_gameboard, player, asset_group)

    # using owned assets
    elif action == "improve_property":
        return get_action_vector_mapping__improve_property(current_gameboard, player, asset_group)

    # using owned assets
    elif action == "make_trade_offer":
        return get_action_vector_mapping__make_trade_offer(current_gameboard, player, asset_group)

    else:
        return (None, None, True)


def get_reward_v1(game_elements, player):
    """
    Dense reward function - Bailis et. al.
    https://github.com/pmpailis/rl-monopoly/blob/master/csharp/Monopoly/Monopoly/RLHandlers/RLEnvironment.cs
    """
    # smoothing factor
    # change this as desired. keep the result between -1 and +1
    c = 5

    # (value sum of all owned assets) - (value sum of all assets owned by other players)
    # keep the value 0 here. it is calculated later.
    v = 0
    total_cash = 0
    # (player's money) / (all players' money)
    # keep the value 0 here. it is calculated later.
    m = 0

    # -----
    # Some points to keep in mind
    # 1. Utility and Railroad get lower reward than if it is a realty property - utility and railroad don't make
    #    you as much money

    active_players = get_active_players(game_elements)
    num_active_players = len(active_players)
    for pl in active_players:
        # sum of all active players' cash
        total_cash += pl.current_cash
        # higher reward for full_color_set_possessed
        if pl == player:
            v += 4 * len(pl.full_color_sets_possessed)
        else:
            v -= 4 * len(pl.full_color_sets_possessed)

    # sum of assets owned by player
    if len(player.assets) > 0:
        for asset in get_all_assets_of_player(player):
            v += get_asset_value_v1(asset)

    # sum of assets owned by other players
    for asset in get_assets_owned_by_other_players(game_elements, player):
        v -= get_asset_value_v1(asset)

    # this player's cash / total players cash
    if total_cash > 0:
        m = player.current_cash / total_cash

    # sigmoid function
    part1 = (v / num_active_players * c)
    part2 = 1 + abs(part1)
    part3 = m

    r = (part1 / part2) + part3
    return torch.tensor([r])


def get_reward_v2(game_elements, player):
    # number of players
    p = len(game_elements['players'])

    # smoothing factor
    # change this as desired. keep the result between -1 and +1
    c = 0.005

    # (value sum of all owned assets) - (value sum of all assets of other players)
    # keep the value 0 here. it is calculated later.
    v = 0

    # (player's money) / (all players' money)
    # keep the value 0 here. it is calculated later.
    m = 0

    # -----

    # sum of value of owned assets
    for asset in get_all_assets_of_player(player):
        if isinstance(asset, location.RailroadLocation):
            v += asset.price

        elif isinstance(asset, location.UtilityLocation):
            v += asset.price

        elif isinstance(asset, location.RealEstateLocation):
            # one house and one hotel are same price in Monopoly.
            v += asset.price + (asset.num_houses * asset.price_per_house) + (asset.num_hotels * asset.price_per_house)

    # for other players
    for other_player in game_elements['players']:
        if other_player is player:
            continue

        for asset in get_all_assets_of_player(other_player):
            if isinstance(asset, location.RailroadLocation):
                v -= asset.price

            elif isinstance(asset, location.UtilityLocation):
                v -= asset.price

            elif isinstance(asset, location.RealEstateLocation):
                # one house and one hotel are same price in Monopoly.
                v -= asset.price + (asset.num_houses * asset.price_per_house) + (
                        asset.num_hotels * asset.price_per_house)

    # -----

    # sum up cash of all players
    for other_player in game_elements['players']:
        m += other_player.current_cash

    # (this player's cash) / (other players' cash)
    m = player.current_cash / m

    # -----

    # sigmoid function
    part1 = (v * c) / p
    part2 = 1 + abs(part1)
    part3 = (m / p)

    r = (part1 / part2) + part3

    return torch.tensor([r])


def calculate_worth_of_player(player):
    # reward function = (current cash) + (value of all properties).
    # Set the first equation to zero to ignore the cash.

    # set this to 0 if current cash is not desired in the result
    value = player.current_cash

    for asset in player.assets:
        asset_group = get_asset_group(asset)
        if asset_group is None:
            continue

        if asset.is_mortgaged:
            continue

        if isinstance(asset, location.RealEstateLocation):
            value += asset.price
            value += (asset.num_houses * asset.price_per_house)
            value += (asset.num_hotels * asset.price_per_house)

        elif isinstance(asset, location.RailroadLocation):
            value += asset.price

        elif isinstance(asset, location.UtilityLocation):
            value += asset.price

    return value


def calculate_worth_of_player2(current_gameboard, pl):
    networth_player = 0
    networth_player += pl.current_cash
    if pl.assets:
        for prop in pl.assets:
            if prop.loc_class == 'real_estate':
                networth_player += prop.price
                networth_player += prop.num_houses*prop.price_per_house
                networth_player += prop.num_hotels*prop.price_per_house*(current_gameboard['bank'].house_limit_before_hotel + 1)
            elif prop.loc_class == 'railroad':
                networth_player += prop.price
            elif prop.loc_class == 'utility':
                networth_player += prop.price

    return networth_player


def get_reward_v3(game_elements, player):
    # NEED TO UPDATE THE FUNCTION
    k = 0.001
    # print(k * calculate_worth_of_player(player), k * calculate_worth_of_player2(game_elements, player))
    return torch.tensor([k * calculate_worth_of_player2(game_elements, player)])


def calculate_worth_of_player3(current_gameboard, player):
    others_networth = 0
    networth_current_player = 0
    for pl in current_gameboard['players']:   # in case there are no winners, we find the winner as the player with highest networth
        if pl.status != 'lost':
            networth_player = 0
            networth_player += pl.current_cash
            if pl.assets:
                for prop in pl.assets:
                    if prop.loc_class == 'real_estate':
                        networth_player += prop.price
                        networth_player += prop.num_houses*prop.price_per_house
                        networth_player += prop.num_hotels*prop.price_per_house*(current_gameboard['bank'].house_limit_before_hotel + 1)
                    elif prop.loc_class == 'railroad':
                        networth_player += prop.price
                    elif prop.loc_class == 'utility':
                        networth_player += prop.price
            if pl == player:
                networth_current_player += networth_player
            else:
                others_networth += networth_player
    return torch.tensor([networth_current_player/others_networth])


def get_reward_v5(game_elements, player):
    # print(k * calculate_worth_of_player(player), k * calculate_worth_of_player2(game_elements, player))
    return calculate_worth_of_player3(game_elements, player)

# -----


def is_player_active(player):
    return not ((player.status == 'won') or (player.status == 'lost'))


def get_active_players(game_elements):
    active_players = []

    for player in game_elements['players']:
        if is_player_active(player):
            active_players.append(player)

    return active_players


def get_all_asset_groups(game_elements):
    asset_groups = []

    for color_asset in game_elements['color_assets']:
        asset_groups.append(color_asset)

    asset_groups.append("Railroad")
    asset_groups.append("Utility")

    return asset_groups


def get_asset_group(asset):
    is_realestate = isinstance(asset, location.RealEstateLocation)
    is_railroad = isinstance(asset, location.RailroadLocation)
    is_utility = isinstance(asset, location.UtilityLocation)

    if is_railroad:
        return "Railroad"

    elif is_utility:
        return "Utility"

    elif is_realestate:
        return asset.color

    else:
        return None


def get_assets_of_player_by_group(player, desired_asset_group):
    assets_of_player = []

    if player.assets is not None:
        for asset in player.assets:
            if get_asset_group(asset) == desired_asset_group:
                assets_of_player.append(asset)

    return assets_of_player


def get_all_assets(game_elements):
    assets = []

    for obj_name in game_elements['location_objects']:
        loc = game_elements['location_objects'][obj_name]

        # some location objects do not have any useful information. thus, we skip them
        if isinstance(loc, location.RealEstateLocation):
            assets.append(loc)

        elif isinstance(loc, location.RailroadLocation):
            assets.append(loc)

        elif isinstance(loc, location.UtilityLocation):
            assets.append(loc)

    return assets


def get_all_assets_of_player(player):
    assets_of_player = []

    if player.assets is not None:
        for asset in player.assets:
            if get_asset_group(asset) != None:
                assets_of_player.append(asset)

    return assets_of_player


def get_all_assets_not_of_player(game_elements, player):
    assets = []

    for obj_name in game_elements['location_objects']:
        loc = game_elements['location_objects'][obj_name]

        # some location objects do not have any useful information. thus, we skip them
        if isinstance(loc, location.RealEstateLocation):
            if loc.owned_by != player:
                assets.append(loc)

        elif isinstance(loc, location.RailroadLocation):
            if loc.owned_by != player:
                assets.append(loc)

        elif isinstance(loc, location.UtilityLocation):
            if loc.owned_by != player:
                assets.append(loc)

    return assets


def get_assets_owned_by_other_players(game_elements, player):
    assets = []

    for obj_name in game_elements['location_objects']:
        loc = game_elements['location_objects'][obj_name]
        if isinstance(loc, location.RealEstateLocation) or \
                isinstance(loc, location.RailroadLocation) or \
                isinstance(loc, location.UtilityLocation):
            if not isinstance(loc.owned_by, bank.Bank) and \
                    not loc.is_mortgaged and \
                    loc.owned_by != player:
                assets.append(loc)

    return assets


def get_asset_value_v1(asset):
    # Used with get_reward_v1
    value = 0

    if isinstance(asset, location.RailroadLocation) and not asset.is_mortgaged:
        value += 1

    elif isinstance(asset, location.UtilityLocation) and not asset.is_mortgaged:
        value += 1

    elif isinstance(asset, location.RealEstateLocation) and not asset.is_mortgaged:
        # one house and one hotel are same price in Monopoly.
        # give real estate a higher value
        value += (2 + asset.num_houses + 2 * asset.num_hotels)
    return value
