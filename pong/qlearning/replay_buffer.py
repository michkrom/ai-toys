"""
Experience replay buffer for stable Q-learning with Prioritized Replay support.

Stores (state, action, reward, next_state, done) tuples and allows
sampling random batches for training. With prioritized replay, samples
based on TD error magnitude for more efficient learning.
"""

import random
from collections import deque
from typing import List
import math


class ReplayBuffer:
    """
    Experience replay buffer for stable Q-learning.
    Supports both uniform and prioritized experience replay.
    """

    def __init__(
        self, 
        capacity: int = 10000, 
        prioritized: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6
    ):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha  # Prioritization exponent (0 = uniform, 1 = fully prioritized)
        self.beta = beta    # Importance sampling exponent
        self.eps = eps      # Small constant for numerical stability
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
        info: dict = None,
        priority: float = None,
    ):
        """Add an experience to the buffer.

        priority: explicit sampling priority (e.g. the TD error computed by
        the agent at store time). Defaults to the current max priority so
        fresh experiences are always at least sampled once.
        """
        if priority is None:
            priority = self.max_priority

        experience = (
            state,
            action,
            reward,
            next_state,
            done,
            info or {}
        )
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.prioritized and len(self.priorities) > 0:
            # Prioritized sampling based on priority^alpha
            priorities = [p ** self.alpha for p in self.priorities]
            total = sum(priorities)
            probs = [p / total for p in priorities]

            # Sample with weights
            indices = random.choices(
                range(len(self.buffer)),
                weights=probs,
                k=batch_size
            )

            # Importance sampling weights (one per SAMPLED item, aligned
            # with `indices`; previously computed for all entries)
            weights = [
                (len(self.buffer) * probs[i]) ** (-self.beta)
                for i in indices
            ]
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]

            batch = [self.buffer[i] for i in indices]
            return batch, indices, weights
        else:
            # Uniform sampling
            indices = random.sample(range(len(self.buffer)), batch_size)
            batch = [self.buffer[i] for i in indices]
            weights = [1.0] * batch_size
            return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        if not self.prioritized:
            return
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = max(priority, self.eps)
                self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self) -> int:
        return len(self.buffer)