import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from gymnasium.wrappers import FrameStackObservation

import config


# -------------------
# Auxiliary functions
# -------------------

# maps agent action representation to environment action representation:
#   - actions: 0 -> 0 (NOOP); 1 -> 3 (LEFT); 2 -> 2 (RIGHT)
parse_agent_action = lambda agent_agent: config.AGENT2ENV_ACTION[agent_agent]

format_step_idx = lambda step_idx: str(step_idx).zfill(len(str(config.NUM_TRAINING_STEPS)))

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
        env_action = config.AGENT2ENV_ACTION[action]

        next_state, _, terminated, truncated, _ = env.step(env_action)
        
        done = terminated or truncated
        state = preprocess_state(next_state)