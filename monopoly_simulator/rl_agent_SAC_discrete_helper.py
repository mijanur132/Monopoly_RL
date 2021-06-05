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


        "learning_rate": 0.0005,
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
            "learning_rate": 0.001,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.01,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "batch_size": 50,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0

}

