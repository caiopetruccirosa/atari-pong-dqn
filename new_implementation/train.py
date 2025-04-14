import gymnasium as gym
import torch.optim as optim
import ale_py
import random
import pickle
import signal 
import sys

from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

import config

from replay_buffer import ReplayBuffer
from agent import DQNAgent
from utils import *

# global variable to indicate to stop training
STOP_TRAINING = False

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
    optimizer = optim.Adam(agent.policy_network.parameters(), lr=config.LR)
    epsilon = config.EPSILON_START

    agent.update_target_network()

    history = {
        'steps_per_episode': [],
        'reward_per_episode': [],
    }

    pbar = tqdm(total=config.NUM_TRAINING_STEPS)

    step = 0
    while step < config.NUM_TRAINING_STEPS and not STOP_TRAINING:
        done = False
        ep_step_count = 0
        ep_reward = 0

        state, _ = env.reset()
        state = preprocess_state(state)
        while not done and not STOP_TRAINING:
            if step < config.NUM_RANDOM_POLICY_STEPS:
                action = random.randint(0, config.ACTION_DIM-1)
            else:
                action = agent.choose_action(state, epsilon)
            env_action = config.AGENT2ENV_ACTION[action]

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
            if step % config.UPDATE_POLICY_FREQUENCY == 0 and len(replay_buffer) >= config.BATCH_SIZE:
                transitions_batch = preprocess_transitions(replay_buffer.sample(config.BATCH_SIZE))
                agent.optimize_policy(optimizer, transitions_batch)
                
            # copies policy network state to target network state
            if step % config.UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_network()

            # linear annealing of epsilon, similar to original paper
            if step >= config.NUM_RANDOM_POLICY_STEPS:
                decaying_steps = step-config.NUM_RANDOM_POLICY_STEPS
                epsilon_decay_offset = (config.EPSILON_START-config.EPSILON_END)*(decaying_steps/config.NUM_EPSILON_DECAY_STEPS)
                epsilon = max(config.EPSILON_START-epsilon_decay_offset, config.EPSILON_END)

            step += 1
            ep_step_count += 1
            ep_reward += reward
            state = next_state

            if step < config.NUM_TRAINING_STEPS:
                pbar.update()

        if verbose:
            print(f"[Step {format_step_idx(step)} / {config.NUM_TRAINING_STEPS}] \t Episode Reward: \t {ep_reward}, \t Episode Steps: {ep_step_count}, \t Epsilon: {epsilon:.2f}")

        if done:
            history['steps_per_episode'].append(ep_step_count)
            history['reward_per_episode'].append(ep_reward)

    return agent, history

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
    agent = DQNAgent(device)

    env = make_environment()
    agent, training_history = train_agent(agent, env, verbose=True)
    env.close()

    with open('dqn_agent_checkpoint_mlp.pkl', 'wb') as f:
        pickle.dump(agent, f)

    plot_history(
        history=training_history, 
        attribute='reward_per_episode',
        title='History of Reward per Episode for DQN Agent', 
        xlabel='Episodes Played', 
        ylabel='Reward', 
        fig_path='reward_per_ep_plot_mlp.jpg',
    )

    plot_history(
        history=training_history, 
        attribute='steps_per_episode',
        title='History of Steps per Episode for DQN Agent', 
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