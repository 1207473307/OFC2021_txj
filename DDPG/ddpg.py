import torch
import torch.nn.functional as functional
from torch import optim
from DDPG.Base_Agent import Base_Agent
from DDPG.Replay_Buffer import Replay_Buffer
from DDPG.OU_Noise_Exploration import OU_Noise_Exploration
from DDPG.GNN import Net
import numpy as np
import visdom

class DDPG(Base_Agent):
    """A DDPG Agent"""
    agent_name = "DDPG"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.hyperparameters = config.hyperparameters
        # self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        # self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_local = Net(config.state_dim_1, config.state_dim_2, config.hidden_size, 1, action_dim=1)
        self.critic_target = Net(config.state_dim_1, config.state_dim_2, config.hidden_size, 1, action_dim=1)
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed)
        # self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        # self.actor_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_local = Net(config.state_dim_1, config.state_dim_2, config.hidden_size, config.action_dim)
        self.actor_target = Net(config.state_dim_1, config.state_dim_2, config.hidden_size, config.action_dim)
        Base_Agent.copy_model_over(self.actor_local, self.actor_target)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.exploration_strategy = OU_Noise_Exploration(self.config)

        self.viz = visdom.Visdom()


    def step(self):
        """Runs a step in the game"""
        while not self.done:
            # print("State ", self.state.shape)
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    states, actions, rewards, next_states, dones = self.sample_experiences()
                    self.critic_learn(states, actions, rewards, next_states, dones)
                    self.actor_learn(states)
            self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            self.global_step_number += 1
        self.episode_number += 1

    def sta_0_1(self, action):
        num_0 = 0
        num_1 = 0
        for a in action:
            if a == 0:
                num_0 += 1
            else:
                num_1 += 1
        return [num_0, num_1]

    def learn(self):
        # if self.time_for_critic_and_actor_to_learn():
        #for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
        states, actions, rewards, next_states, dones = self.sample_experiences()
        c_loss = self.critic_learn(states, actions, rewards, next_states, dones)
        a_loss = self.actor_learn(states)
        self.global_step_number += 1

        self.viz.line(Y=[c_loss.detach().numpy()], X=[self.global_step_number], win='c_loss', update='append', opts=dict(title='c_loss', legend=['c_loss']))
        self.viz.line(Y=[a_loss.detach().numpy()], X=[self.global_step_number], win='a_loss', update='append', opts=dict(title='a_loss', legend=['a_loss']))
        self.viz.line(Y=[torch.mean(rewards)], X=[self.global_step_number], win='reward', update='append', opts=dict(title='reward', legend=['reward']))
        self.viz.line(Y=[self.sta_0_1(actions)], X=[self.global_step_number], win='action', update='append', opts=dict(title='action', legend=['0', '1']))

    def sample_experiences(self):
        return self.memory.sample()

    def get_action(self, state):
        with torch.no_grad():
            probs = self.actor_local(state)#.cpu().data.numpy()
        #self.actor_local.train()
        probs = functional.softmax(probs, dim=0)
        prob, act_id = torch.topk(probs, 1, dim=0)
        action = act_id
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        if action >= 0.5:
            return 1
        else:
            return 0

    def get_action_target(self, state):
        with torch.no_grad():
            probs = self.actor_target(state)  # .cpu().data.numpy()
        # self.actor_local.train()
        probs = functional.softmax(probs, dim=0)
        prob, act_id = torch.topk(probs, 1, dim=0)
        action = act_id
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        if action >= 0.5:
            return 1
        else:
            return 0

    def pick_action(self, state=None):
        """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
        if state is None: state = torch.from_numpy(self.state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action = self.exploration_strategy.perturb_action_for_exploration_purposes({"action": action})
        return action.squeeze(0)

    def critic_learn(self, states, actions, rewards, next_states, dones):
        """Runs a learning iteration for the critic"""
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, loss, self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyperparameters["Critic"]["tau"])

        return loss

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss for the critic"""
        with torch.no_grad():
            critic_targets = self.compute_critic_targets(next_states, rewards, dones)
        critic_expected = self.compute_expected_critic_values(states, actions)
        loss = functional.mse_loss(critic_expected, critic_targets)
        return loss

    def compute_critic_targets(self, next_states, rewards, dones):
        """Computes the critic target values to be used in the loss for the critic"""
        critic_targets_next = self.compute_critic_values_for_next_states(next_states)
        critic_targets = self.compute_critic_values_for_current_states(rewards, critic_targets_next, dones)
        return critic_targets

    def compute_critic_values_for_next_states(self, next_states):#
        """Computes the critic values for next states to be used in the loss for the critic"""
        # with torch.no_grad():
        #     actions_next = self.actor_target(next_states)
        #     critic_targets_next = self.critic_target(torch.cat((next_states, actions_next), 1))
        with torch.no_grad():
            critic_targets_next = []
            for next_s in next_states:
                #actions_next = self.actor_target(next_s)
                actions_next = torch.tensor(self.get_action_target(next_s)).unsqueeze(0)
                critic_targets_next.append(self.critic_target(next_s, action=actions_next))
            critic_targets_next = torch.from_numpy(np.vstack(critic_targets_next))

        return critic_targets_next

    def compute_critic_values_for_current_states(self, rewards, critic_targets_next, dones):
        """Computes the critic values for current states to be used in the loss for the critic"""
        critic_targets_current = rewards + (self.hyperparameters["discount_rate"] * critic_targets_next * (1.0 - dones))
        return critic_targets_current

    def compute_expected_critic_values(self, states, actions):#
        """Computes the expected critic values to be used in the loss for the critic"""
        #critic_expected = self.critic_local(torch.cat((states, actions), 1))

        critic_expected = []
        for s,a in zip(states, actions):
            critic_expected.append(self.critic_local(s, action=a))
        critic_expected = torch.cat(critic_expected).unsqueeze(1)
        return critic_expected


    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def actor_learn(self, states):
        """Runs a learning iteration for the actor"""
        # if self.done: #we only update the learning rate at end of each episode
        #     self.update_learning_rate(self.hyperparameters["Actor"]["learning_rate"], self.actor_optimizer)
        actor_loss = self.calculate_actor_loss(states)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.actor_local, self.actor_target, self.hyperparameters["Actor"]["tau"])

        return actor_loss

    def calculate_actor_loss(self, states):#
        """Calculates the loss for the actor"""
        # actions_pred = self.actor_local(states)
        # actor_loss = -self.critic_local(torch.cat((states, actions_pred), 1)).mean()
        actor_loss = []
        for s in states:
            action_pred = torch.tensor(self.get_action(s)).unsqueeze(0)
            actor_loss.append(-self.critic_local(s, action=action_pred))
        actor_loss = torch.cat(actor_loss).mean()
        return actor_loss