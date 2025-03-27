import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.core import ObsType

import ale_py

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import random

from tqdm import tqdm

from collections import deque


# Pong Environment:
#   - wraps Gymnasium/ALE environment for parsing action and state representations
class PongEnvironment:
    def __init__(self, render_mode=None, is_video_recording:bool=False, video_dir:str|None=None, video_filename:str|None=None):
        gym_env = gym.make("ALE/Pong-v5", obs_type="ram", render_mode=render_mode, full_action_space=False)
        if not is_video_recording:
            self.env = gym_env
        else:
            self.env = RecordVideo(gym_env.env, video_folder=video_dir, name_prefix=video_filename, episode_trigger=lambda _: True)

    def __parse_ram_state(self, ram_state: ObsType) -> np.ndarray:
        # indexes for values on RAM array found on:
        #   https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
        state = np.array([
            ram_state[51], # 'player_y' value
            ram_state[50], # 'enemy_y' value
            ram_state[49], # 'ball_x' value
            ram_state[54], # 'ball_y' value
            ram_state[13], # 'enemy_score' value
            ram_state[14], # 'player_score' value
        ], dtype=np.float32)
        state = state / 255.0
        return state
    
    def __parse_agent_action(self, agent_action: int) -> int:
        # maps agent action representation to environment action representation:
        #   0 -> 0 (NOOP), 1 -> 3 (LEFT), 2 -> 2 (RIGHT)
        agent2env_action = [0, 3, 2]
        return agent2env_action[agent_action]

    def reset(self) -> np.ndarray:
        state, _ = self.env.reset()
        state = self.__parse_ram_state(state)
        return state

    def step(self, agent_action: int) -> tuple[np.ndarray, float, bool]:
        action = self.__parse_agent_action(agent_action)
        
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        next_state = self.__parse_ram_state(next_state)
        finished = int(terminated or truncated)

        return (next_state, reward, finished)
    
    def close(self):
        self.env.close()


# Deep Q-Network:
#   - implementation of a simple MLP network
#   - takes state observation and return actions values as a Q-function approximation
class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        o = self.relu(self.fc1(x))
        o = self.relu(self.fc2(o))
        o = self.fc3(o)
        return o


# Replay Buffer:
#   - stores a circular buffer, using a deque, of environment transitions and their rewards
class ReplayBuffer:
    def __init__(self, buffer_capacity: int):
        self.buffer_capacity = buffer_capacity
        self.buffer = deque(maxlen=buffer_capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, finished: int):
        self.buffer.append((state, action, reward, next_state, finished))
    
    def sample_batch(self, batch_size: int) -> tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], list[int]]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, finished = zip(*batch)
        state = torch.tensor(np.array(state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        finished = torch.tensor(np.array(finished), dtype=torch.float)
        return state, action, reward, next_state, finished
    
    def __len__(self) -> int:
        return len(self.buffer)


# Deep Q-Network Agent: 
#   - chooses action based on state observation using policy network
#   - trains the policy network of the agent in a pong environment using DQN learning with experience replay
class DQNAgent:
    def __init__(self, state_dim: int,  action_dim: int, device: torch.device):
        self.device = device
        
        self.epsilon = 0

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.policy_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNetwork(self.state_dim, self.action_dim).to(self.device)

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
       
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        
        action_values = self.policy_network(state)
        choosen_action = action_values.argmax().item()
        
        return choosen_action

    def optimize_policy(
        self,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        gamma: float,
    ):
        if len(replay_buffer) < batch_size:
            return
        
        loss_func = nn.MSELoss()
        
        state_t, action_t, reward_t, next_state_t, finished_t = replay_buffer.sample_batch(batch_size)
        state_t = state_t.to(self.device)
        action_t = action_t.to(self.device)
        reward_t = reward_t.to(self.device)
        next_state_t = next_state_t.to(self.device)
        finished_t = finished_t.to(self.device)
        
        # select q-value for the action taken in the current state:
        #   policy_network predicts q-value for all possible actions,
        #   and gather() selects the q-value predicted for the action taken
        # obs: unsqueeze() just adds another dimension for gather operation and squeeze() removes dimension after operation
        current_action_value = self.policy_network(state_t).gather(1, action_t.unsqueeze(1)).squeeze()

        # get maximum q-value for next state using target network
        next_action_value = self.target_network(next_state_t).amax(dim=1)

        # calculate target q-value according to bellman optimality equation
        target_action_value = reward_t + (1-finished_t) * gamma * next_action_value
        
        # compute loss between predicted q-values by the policy network and target q-values based on target network
        # obs: detach() prevents gradients from flowing through target network
        loss = loss_func(current_action_value, target_action_value.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(
        self, 
        env: PongEnvironment, 
        n_episodes: int, 
        batch_size: int, 
        lr: float, 
        gamma: float, 
        epsilon_start: float, 
        epsilon_end: float, 
        epsilon_decay: float,
        update_target_network_after_n_steps: int, 
        replay_buffer_capacity: int,
        verbose:bool=False,
    ) -> list[float]:
        format_epsidode_idx = lambda ep_idx: str(ep_idx).zfill(len(str(n_episodes)))

        optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # initializes target network to current policy network and epsilon for training
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.epsilon = epsilon_start

        replay_buffer = ReplayBuffer(replay_buffer_capacity)
        acc_reward_history = []
        
        for episode_idx in tqdm(range(n_episodes)):
            finished_current_ep = False
            step = 0
            acc_reward = 0

            state = env.reset()
            while not finished_current_ep:
                action = self.choose_action(state)
                next_state, reward, finished_current_ep = env.step(action)
                
                replay_buffer.push(state, action, reward, next_state, finished_current_ep)
                
                self.optimize_policy(optimizer, replay_buffer, batch_size, gamma)

                state = next_state
                acc_reward += reward
            
                # copies policy network state to target network state
                if step % update_target_network_after_n_steps == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                step += 1

            # updates epsilon according to decay until reaches its final value
            self.epsilon = max(self.epsilon*epsilon_decay, epsilon_end)
            
            acc_reward_history.append(acc_reward)

            if verbose:
                print(f"[Episode {format_epsidode_idx(episode_idx)}] \t Reward: \t {acc_reward}, \t Steps: {step}, \t Epsilon: {self.epsilon:.2f}")

        return acc_reward_history
    

class ResultsReport:
    def plot_accumulated_reward_history(acc_reward_history: list[float], fig_path: str):
        plt.plot(acc_reward_history)
        plt.title('Accumulated Reward History for DQN Agent')
        plt.xlabel('Episodes Played')
        plt.ylabel('Total Accumulated Reward')
        plt.savefig(fig_path)
        plt.close()

    def record_agent_playing(env: PongEnvironment, agent: DQNAgent):
        state = env.reset()
        finished_current_ep = False
        while not finished_current_ep:
            action = agent.choose_action(state)
            next_state, _, finished_current_ep = env.step(action)
            state = next_state
        env.close()


def main():
    gym.register_envs(ale_py)
    
    random.seed(42)
    torch.manual_seed(42)

    # state representation will be an array of:
    #   [ player_y, enemy_y, ball_x, ball_y, enemy_score, player_score ]
    # action representation will be an array of:
    #   [ NOOP, LEFT, RIGHT ]
    state_dim = 6
    action_dim = 3

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    )

    training_env = PongEnvironment()
    acc_reward_history = agent.train(
        env=training_env,
        n_episodes=5000,
        batch_size=64,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        update_target_network_after_n_steps=1000,
        replay_buffer_capacity=100000,
        verbose=True,
    )
    training_env.close()

    ResultsReport.plot_accumulated_reward_history(acc_reward_history, 'plot.jpg')

    video_env = PongEnvironment(render_mode="rgb_array", is_video_recording=True, video_dir='.', video_filename='video')
    ResultsReport.record_agent_playing(video_env, agent)
    video_env.close()


if __name__ == "__main__":
    main()