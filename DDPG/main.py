import gym
from DDPG.Trainer import Trainer
from DDPG.ddpg import DDPG
from DDPG.Config import Config
from DDPG.Env import Multicast_Env
from Net_Graph import G
import torch

config = Config()
config.seed = 0
config.environment = Multicast_Env
config.num_episodes_to_run = 1500
config.file_to_save_data_results = None
config.file_to_save_results_graph = None
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = torch.cuda.is_available()
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.G = G
config.batch_size = 32
config.state_dim_1 = 3
config.state_dim_2 = 2
config.action_dim = 2
config.hidden_size = 128
config.action_size = 1
config.path = path = '/home/txj/OFC2021/OFC2021_txj'
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config.hyperparameters = {
    "Actor": {
        "learning_rate": 0.001,
        "linear_hidden_units": [400, 300],
        "final_layer_activation": "TANH",
        "batch_norm": False,
        "tau": 0.01,
        "gradient_clipping_norm": 5
    },

    "Critic": {
        "learning_rate": 0.01,
        "linear_hidden_units": [400, 300],
        "final_layer_activation": "None",
        "batch_norm": False,
        "buffer_size": 100000,
        "tau": 0.01,
        "gradient_clipping_norm": 5
    },

    "batch_size": 64,
    "discount_rate": 0.99,
    "mu": 0.0,  # for O-H noise
    "theta": 0.15,  # for O-H noise
    "sigma": 0.2,  # for O-H noise
    "action_noise_std": 0.2,  # fIntel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHzor TD3
    "action_noise_clipping_range": 0.5,  # for TD3
    "update_every_n_steps": 1,
    "learning_updates_per_learning_session": 1,
    "clip_rewards": False

    }


if __name__ == "__main__":
    # AGENTS = [DDPG]
    # trainer = Trainer(config, AGENTS)
    # trainer.run_games_for_agents()
    agent = DDPG(config)
    Env = Multicast_Env(config, agent)
    Env.run()




