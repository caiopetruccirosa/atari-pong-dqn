from collections import deque

import config


# -------------
# Replay Buffer
# -------------
class ReplayBuffer:
    def __init__(self):
        self.buffer = deque([], maxlen=config.REPLAY_BUFFER_CAPACITY)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)

    def __len__(self):
        return len(self.buffer)