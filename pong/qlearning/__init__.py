"""
Q-Learning with Deep Neural Network for Pong AI

This package provides:
- QNetwork: Neural network for Q-value estimation
- ReplayBuffer: Experience replay buffer
- QLearningAgent: Self-learning AI controller
- SelfPlayTrainer: Self-play training utilities
"""

from .network import QNetwork
from .replay_buffer import ReplayBuffer
from .agent import QLearningAgent
from .trainer import SelfPlayTrainer, train_against_perfect, train_quick

__all__ = [
    "QNetwork",
    "ReplayBuffer",
    "QLearningAgent",
    "SelfPlayTrainer",
    "train_against_perfect",
    "train_quick",
]
