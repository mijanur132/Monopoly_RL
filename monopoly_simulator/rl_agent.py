import torch
import random
import torch.optim as optim
from monopoly_simulator.agent import Agent
from monopoly_simulator.rl_agent_helper import DQN, EpsilonGreedyStrategy, ReplayMemory
from monopoly_simulator import location, rl_agent_helper

import math
from actor_critic_agents.SAC_Discrete import SAC_Discrete

from actor_critic_utilities.data_structures.Config import Config
import random
import torch

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


def get_act_loc_map(current_gameboard):
    loc_type_set = set()
    final_loc_type_list = []
    loc_action_map = {}
    action_type_list = ['sell', 'spend']
    m_num = len(action_type_list)
    for obj_name in current_gameboard['location_objects']:
        loc = current_gameboard['location_objects'][obj_name]
        if isinstance(loc, location.RealEstateLocation):
            loc_type_set.add(loc.color)
        elif isinstance(loc, location.RailroadLocation):
            loc_type_set.add('Railroad')
        elif isinstance(loc, location.UtilityLocation):
            loc_type_set.add('Utility')
    loc_type_set_list = sorted(list(loc_type_set))
    for loc_type in loc_type_set_list:
        final_loc_type_list.append(loc_type + '_sell')
        final_loc_type_list.append(loc_type + '_spend')
        # final_loc_type_list.append(loc_type + '_do_nothing')

    for i in range(len(loc_type_set_list)):
        for j, k in enumerate(range(i * m_num, i * m_num + m_num)):
            loc_action_map[k] = {'loc_group': loc_type_set_list[i],
                                 'action_type': action_type_list[j]}
    # print(loc_action_map)
    # print(k)
    loc_action_map[k + 1] = {'loc_group': None,
                             'action_type': 'do_nothing'
                             }
    return loc_action_map


class RLAgent(Agent):
    def __init__(self, handle_negative_cash_balance, make_pre_roll_move,
                 make_out_of_turn_move,
                 make_post_roll_move, make_buy_property_decision, make_bid, type, load_path, train=True, eps=0.1):
        super().__init__(handle_negative_cash_balance, make_pre_roll_move,
                         make_out_of_turn_move,
                         make_post_roll_move, make_buy_property_decision, make_bid, type)


        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 1
        self.eps_end = 0.01
        self.eps_decay = 0.000001
        self.target_update = 1000
        self.checkpoint = 10000
        self.memory_size = 100000
        self.lr = 0.00001
        # self.num_episodes = 1000
        self.action_map, self.action_map_r = {}, {}
        self.act_loc_map = None
        self.input_dim = 23  # 277 for state_vector, 23 for state_vector2
        self.output_dim = 66  # Hard-coded to action space. 21 for action1, 76 for action2

        self.current_step = 0
        self.num_actions = self.output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = ReplayMemory(self.memory_size)

        # Initialize the policy and target networks
        # self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        # self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.agent_sac=SAC_Discrete(config)
        # Set the optimizer
        #self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        self.loss = None

        # Things to push onto the memory - let's keep them with the agent so that we can push at any instance:
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.current_game_count = 0

        if load_path is not None and train:
            checkpoint = torch.load(load_path)
            # self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss = checkpoint['loss']
            self.current_step = checkpoint['current_step']
            self.state = checkpoint['state']
            self.action = checkpoint['action']
            self.reward = checkpoint['reward']
            self.next_state = checkpoint['next_state']
            self.memory = checkpoint['replay_memory']
            self._agent_memory['previous_action'] = ''
            self.current_game_count = checkpoint['current_game_count']
        elif load_path is not None:
            # checkpoint = torch.load(load_path, map_location=self.device)
            # self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            # self.policy_net.eval()
            self.eps_start = eps
            self.eps_end = eps
            self._agent_memory['previous_action'] = ''

        self.strategy = EpsilonGreedyStrategy(self.eps_start, self.eps_end, self.eps_decay)

        # Set the weights and biases in the target_net to be the same as those in the policy_net
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # # Set target_net to eval mode - not used for training, only inference
        # self.target_net.eval()

        self.episodic_rewards = 0
        self.episodic_step = 0
        self.failed_actions = 0
        self.successful_actions = 0


        # self.move_phase = None

    # @staticmethod
    # def get_action_map():
    #     pre_die_roll = [
    #         "mortgage_property",
    #         "improve_property",
    #         "use_get_out_of_jail_card",
    #         "pay_jail_fine",
    #         "skip_turn",
    #         "free_mortgage",
    #         "sell_property",
    #         "sell_house_hotel",  # Separate the sell house and sell hotel actions for the DQN
    #         "accept_sell_property_offer",
    #         "roll_die",
    #         "concluded_actions",
    #         "make_sell_property_offer"
    #     ]
    #     post_die_roll = [
    #         "mortgage_property",
    #         "buy_property",
    #         "sell_property",
    #         "sell_house_hotel",
    #         "concluded_actions"
    #     ]
    #
    #     out_of_turn = [
    #         "free_mortgage",
    #         "sell_property",
    #         "sell_house_hotel",
    #         "accept_sell_property_offer",
    #         "make_sell_property_offer",
    #         "skip_turn",
    #         "concluded_actions",
    #         "mortgage_property",
    #         "improve_property"
    #     ]
    #     other_actions = [
    #         "make_trade_offer",
    #         "accept_trade_offer"
    #     ]
    #     action_set = sorted(set(pre_die_roll + post_die_roll + out_of_turn + other_actions))
    #     # Maps integers to action string
    #     action_map = dict([(x, y) for x, y in enumerate(action_set)])
    #     # Maps action string to integers
    #     action_map_r = dict([(y, x) for x, y in enumerate(action_set)])
    #     return action_map, action_map_r

    def startup(self, current_gameboard, indicator=None):
        val = super(RLAgent, self).startup(current_gameboard, indicator=None)
        self.current_game_count += 1

        self.strategy.rate = self.strategy.end + (self.strategy.start - self.strategy.end) * math.exp(-1. *
                                                                                                          self.current_game_count * self.eps_decay)
        self.act_loc_map = get_act_loc_map(current_gameboard)
        self.action_map = rl_agent_helper.get_action_vector5(current_gameboard)
        for i in self.action_map:
            action = self.action_map[i]['action_type']
            if action not in self.action_map_r:
                self.action_map_r[action] = {i}
            else:
                self.action_map_r[action].add(i)
        # print(self.action_map_r)
        return val
