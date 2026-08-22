"""
Q-Learning with Deep Neural Network for Self-Learning Pong AI

This module provides the main entry point for the Q-learning system.
It imports from the qlearning package and exposes the main classes and functions.

Usage:
    from qlearning_agent import QLearningAgent, QNetwork, train_against_perfect
    
Or run training:
    python train_qlearning.py --episodes 100
"""

# Re-export from qlearning package
from qlearning.agent import QLearningAgent
from qlearning.network import QNetwork
from qlearning.trainer import train_against_perfect

__all__ = ['QLearningAgent', 'QNetwork', 'train_against_perfect']


if __name__ == "__main__":
    print("\nQuick test of Q-Learning agent...")

    agent = QLearningAgent(side="right")

    print("\nTraining for 50 episodes...")
    for episode in range(50):
        agent.train_step()

    print("\nTotal bounces: ", agent.win_count)

    print("\nTesting after training:")
    print("  -> BOUNCED!")
    print("=" * 60)

    print(f"\nFinal epsilon: {agent.epsilon:.3f}")