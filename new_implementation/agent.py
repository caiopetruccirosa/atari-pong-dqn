import torch
import torch.nn.functional as F
import random

import config

from dqn import DQNetwork

# ---------
# DQN Agent
# ---------
class DQNAgent:
    def __init__(self, device):
        self.device = device
        self.policy_network = DQNetwork().to(self.device)
        self.target_network = DQNetwork().to(self.device)

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, config.ACTION_DIM-1)
       
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

        q_action_values = self.policy_network(state)
        choosen_action = q_action_values.argmax().item()
        
        return choosen_action
    
    def optimize_policy(self, optimizer, transitions_batch):
        states, actions, rewards, next_states, dones = transitions_batch

        states = states.to(self.device) # shape (B, 16)
        actions = actions.to(self.device) # shape (B)
        rewards = rewards.to(self.device) # shape (B)
        next_states = next_states.to(self.device) # shape (B, 16)
        dones = dones.to(self.device) # shape (B)
        
        # predicts Q-values for all actions for current state (batched); result of shape (B, ACTION_DIM=3)
        q_values = self.policy_network(states)

        # selects Q-values predicted for action taken (batched); result of shape (B)
        q_value_action_taken = q_values.gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        # predicts Q*-values for all actions in the next states (batched) using the target network; results of shape (B, ACTION_DIM=3)
        next_qs_values = self.target_network(next_states)

        # gets best Q*-value in the next state (batched); result of shape (B)
        best_next_qs_value, _ = next_qs_values.max(dim=1)

        # calculate Q*-value for action taken in current state (batched) according to bellman optimality equation; result of shape (B)
        qs_value_action_taken = rewards + (1-dones) * config.GAMMA * best_next_qs_value
        
        # compute loss between Q-value predicted by the policy network and Q*-value (based on target network) for action taken in current state
        # obs: detach() prevents gradients from flowing through target network
        loss = F.mse_loss(q_value_action_taken, qs_value_action_taken.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()