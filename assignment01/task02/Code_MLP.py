import gymnasium as gym
from gymnasium.wrappers import (
    RecordVideo, 
    FrameStackObservation,
)

import ale_py
import random
import pickle
import signal 
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque


STATE_DIM = 16
ACTION_DIM = 3
AGENT2ENV_ACTION = [0, 3, 2]

EPSILON_START = 1
EPSILON_END = 0.1
NUM_EPSILON_DECAY_STEPS = 500000

BATCH_SIZE = 64
LR = 2.5e-4
GAMMA = 0.99

UPDATE_POLICY_FREQUENCY = 4
UPDATE_TARGET_FREQUENCY = 10000

REPLAY_BUFFER_CAPACITY = 1000000

NUM_RANDOM_POLICY_STEPS = 50000
NUM_TRAINING_STEPS = 5000000

VERBOSE_STEP_FREQUENCY = 10000

STOP_TRAINING = False


# -------------
# Replay Buffer
# -------------
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque([], maxlen=REPLAY_BUFFER_CAPACITY)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        return len(self.buffer)


# --------------
# Deep Q-Network
# --------------
# - uses a MLP as architecture
# - takes state observation and return actions values as a Q-function approximation
class DQNetwork(nn.Module):
    def __init__(self):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=STATE_DIM, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=ACTION_DIM)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        o = self.relu(self.fc1(x))
        o = self.relu(self.fc2(o))
        o = self.fc3(o)
        return o


# ---------
# DDQN Agent
# ---------
class DDQNAgent:
    def __init__(self, device):
        self.device = device
        self.policy_network = DQNetwork().to(self.device)
        self.target_network = DQNetwork().to(self.device)

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, ACTION_DIM-1)
       
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

        # select the best action for next state using policy network (batched); results of shape (B, 1)
        # obs: uses keepdim=True on argmax to maintain shape for gather operation
        best_next_action = self.policy_network(next_states).argmax(dim=1, keepdim=True)

        # predicts Q*-values for all actions in the next states (batched) using the target network; results of shape (B, ACTION_DIM=3)
        next_qs_values = self.target_network(next_states)

        # !!! this is the main difference between DQN and DDQN
        # selects Q*-values for action selected by the policy network (batched); result of shape (B)
        best_next_qs_value = next_qs_values.gather(1, best_next_action).squeeze(1)

        # calculate Q*-value for action taken in current state (batched) according to bellman optimality equation; result of shape (B)
        qs_value_action_taken = rewards + (1-dones) * GAMMA * best_next_qs_value

        # compute loss between Q-value predicted by the policy network and Q*-value (based on target network) for action taken in current state
        # considering action chosen by policy network in the next state when calculating Q*-value
        # obs: uses detach() to prevent gradient from flowing through target network
        loss = F.mse_loss(q_value_action_taken, qs_value_action_taken.detach())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# -------------
# Training loop
# -------------
def train_agent(
    agent,
    env,
    verbose=False,
):
    print(f'Training agent on device \'{agent.device}\'.')

    replay_buffer = ReplayBuffer()
    optimizer = optim.Adam(agent.policy_network.parameters(), lr=LR)
    epsilon = EPSILON_START

    agent.update_target_network()

    history = {
        'steps_per_episode': [],
        'reward_per_episode': [],
    }

    pbar = tqdm(total=NUM_TRAINING_STEPS)

    step = 0
    while step < NUM_TRAINING_STEPS and not STOP_TRAINING:
        done = False
        ep_step_count = 0
        ep_reward = 0

        state, _ = env.reset()
        state = preprocess_state(state)
        while not done and not STOP_TRAINING:
            if step < NUM_RANDOM_POLICY_STEPS:
                action = random.randint(0, ACTION_DIM-1)
            else:
                action = agent.choose_action(state, epsilon)
            env_action = AGENT2ENV_ACTION[action]

            next_state, reward, terminated, truncated, _ = env.step(env_action)
            next_state = preprocess_state(next_state)

            done = terminated or truncated
            replay_buffer.push(
                state=state,
                action=action, 
                reward=reward, 
                next_state=next_state,
                done=done,
            )

            # updates policy network based on experience replay from replay buffer
            if step % UPDATE_POLICY_FREQUENCY == 0 and len(replay_buffer) >= BATCH_SIZE:
                transitions_batch = preprocess_transitions(replay_buffer.sample(BATCH_SIZE))
                agent.optimize_policy(optimizer, transitions_batch)
                
            # copies policy network state to target network state
            if step % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_network()

            # linear annealing of epsilon, similar to original paper
            if step >= NUM_RANDOM_POLICY_STEPS:
                decaying_steps = step-NUM_RANDOM_POLICY_STEPS
                epsilon_decay_offset = (EPSILON_START-EPSILON_END)*(decaying_steps/NUM_EPSILON_DECAY_STEPS)
                epsilon = max(EPSILON_START-epsilon_decay_offset, EPSILON_END)

            step += 1
            ep_step_count += 1
            ep_reward += reward
            state = next_state

            if step < NUM_TRAINING_STEPS:
                pbar.update()

        if verbose:
            print(f"[Step {format_step_idx(step)} / {NUM_TRAINING_STEPS}] \t Episode Reward: \t {ep_reward}, \t Episode Steps: {ep_step_count}, \t Epsilon: {epsilon:.2f}")

        if done:
            history['steps_per_episode'].append(ep_step_count)
            history['reward_per_episode'].append(ep_reward)

    return agent, history


# -------------------
# Auxiliary functions
# -------------------

# maps agent action representation to environment action representation:
#   - actions: 0 -> 0 (NOOP); 1 -> 3 (LEFT); 2 -> 2 (RIGHT)
parse_agent_action = lambda agent_agent: AGENT2ENV_ACTION[agent_agent]

format_step_idx = lambda step_idx: str(step_idx).zfill(len(str(NUM_TRAINING_STEPS)))

def preprocess_state(state):
    # indexes for values on RAM array found on:
    #   https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
    new_state = []
    for frame in state:
        frame_state = [
            frame[51], # 'player_y' value
            frame[50], # 'enemy_y' value
            frame[49], # 'ball_x' value
            frame[54], # 'ball_y' value
        ]
        new_state += frame_state
    new_state = np.array(new_state, dtype=np.float32)
    new_state = new_state / 255.0
    return new_state

def preprocess_transitions(transitions_batch):
    states, actions, rewards, next_states, dones = zip(*transitions_batch)

    # states and next states:
    # - from list[np.ndarray[np.float32]] of shape (B, 4)
    # - to normalized torch.tensor[torch.tensor[torch.float32]] of shape (B, 4)
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

    # rewards:
    # - from from list[float] of shape (B) 
    # - to clipped value torch.tensor[torch.float32] of shape (B)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)

    # actions:
    # - from from list[int] of shape (B) 
    # - to torch.tensor[torch.long] of shape (B)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    
    # dones:
    # - from from list[bool] of shape (B) 
    # - to torch.tensor[torch.float32] of shape (B)
    dones = torch.tensor(np.array(dones), dtype=torch.float32)

    return (states, actions, rewards, next_states, dones)

def make_environment(render_mode=None):
    stacked_frames = 4
    env = gym.make("ALE/Pong-v5", obs_type="ram", render_mode=render_mode, full_action_space=False, frameskip=stacked_frames)
    env = FrameStackObservation(env, stack_size=stacked_frames)
    return env

def plot_history(history, attribute, title, xlabel, ylabel, fig_path):
    plt.plot(history[attribute])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fig_path)
    plt.close()

def record_agent_playing(agent, env):
    state, _ = env.reset()
    state = preprocess_state(state)

    done = False
    while not done:
        action = agent.choose_action(state, epsilon=0.05)
        env_action = AGENT2ENV_ACTION[action]

        next_state, _, terminated, truncated, _ = env.step(env_action)
        
        done = terminated or truncated
        state = preprocess_state(next_state)

def stop_training(signal, frame):
    global STOP_TRAINING
    STOP_TRAINING = True


# ----
# Main
# ----
def main():
    random.seed(42)
    torch.manual_seed(42)

    gym.register_envs(ale_py)

    signal.signal(signal.SIGINT, stop_training)

    if len(sys.argv) > 1:
        device = sys.argv[1]

    device = torch.device(
        sys.argv[1] if len(sys.argv) > 1 else
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    agent = DDQNAgent(device)

    env = make_environment()
    agent, training_history = train_agent(agent, env, verbose=True)
    env.close()

    with open('ddqn_agent_checkpoint_mlp.pkl', 'wb') as f:
        pickle.dump(agent, f)

    plot_history(
        history=training_history, 
        attribute='reward_per_episode',
        title='History of Reward per Episode for DDQN Agent', 
        xlabel='Episodes Played', 
        ylabel='Reward', 
        fig_path='reward_per_ep_plot_mlp.jpg',
    )

    plot_history(
        history=training_history, 
        attribute='steps_per_episode',
        title='History of Steps per Episode for DDQN Agent', 
        xlabel='Episodes Played', 
        ylabel='Number of Steps', 
        fig_path='steps_per_ep_plot_mlp.jpg',
    )

    video_env = make_environment(render_mode="rgb_array")
    video_env = RecordVideo(video_env, video_folder='.', name_prefix='Video-MLP', episode_trigger=lambda _: True)
    record_agent_playing(agent, video_env)
    video_env.close()

if __name__ == "__main__":
    main()