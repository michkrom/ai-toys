"""
Training script for the self-learning Q-Learning AI controller.

Usage:
    python train_agent.py              # Quick training (500 episodes)
    python train_agent.py --episodes 1000  # Longer training
    python train_agent.py --load       # Load and play with trained agent
"""

import argparse
import sys
import os
import time

from qlearning import QLearningAgent
from qlearning.trainer import SelfPlayTrainer, train_against_perfect


def train_agent(episodes=500, opponent_epsilon=0.1):
    """
    Train a Q-learning agent through self-play.

    Args:
        episodes: Number of training episodes
        opponent_epsilon: Opponent exploration rate
    """
    trainer = SelfPlayTrainer(QLearningAgent, episodes=episodes, 
                             opponent_epsilon=opponent_epsilon)
    from physics import PongPhysics
    agent = trainer.train(PongPhysics)
    return agent


def evaluate_agent(agent, episodes=100):
    """
    Evaluate the trained agent against an algorithmic opponent.

    Args:
        agent: Trained QLearningAgent
        episodes: Number of evaluation episodes
    """
    print(f"\nEvaluating agent over {episodes} episodes...")
    print("--------------------------------------------------")

    from controllers import AlgorithmicController
    from physics import PongPhysics

    wins = 0
    losses = 0

    for i in range(episodes):
        physics = PongPhysics()
        state = physics
        action = agent.get_action(state)
        opponent = AlgorithmicController(side="left")
        opponent_action = opponent.get_move(physics)

        physics.update(opponent_action, action)

        if physics.right_score > physics.left_score:
            wins += 1
        else:
            losses += 1

    print(f"Results: {wins} wins, {losses} losses")
    print(f"Win rate: {wins / (wins + losses) if (wins + losses) > 0 else 0:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Train Q-Learning Pong AI')
    parser.add_argument('--method', choices=['selfplay', 'perfect'],
                        default='selfplay',
                        help='Training method: self-play or vs perfect '
                             'controller (default: selfplay)')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--opponent-epsilon', type=float, default=0.1,
                        help='Opponent exploration rate (default: 0.1)')
    parser.add_argument('--load', action='store_true',
                        help='Load trained agent and play')
    parser.add_argument('--evaluate', type=int, default=0,
                        help='Number of evaluation episodes after training')
    parser.add_argument('--save-path', type=str, default='trained_agent.pkl',
                        help='Path to save trained agent')

    args = parser.parse_args()

    if args.load:
        agent = QLearningAgent(side="right")
        if os.path.exists(args.save_path):
            agent.load(args.save_path)
            print(f"Loaded trained agent from {args.save_path}")
        else:
            print(f"No trained agent found at {args.save_path}")
            print("Train first with: python train_agent.py")
            return

        print(f"Final epsilon: {agent.epsilon:.3f}")

        from pong import PongGame
        game = PongGame(left_controller="perfect", right_controller="qlearning")
        game.run()
        return

    if args.method == "perfect":
        from physics import PongPhysics
        from qlearning.trainer import train_against_perfect
        agent = train_against_perfect(PongPhysics, episodes=args.episodes)
    else:
        agent = train_agent(episodes=args.episodes, opponent_epsilon=args.opponent_epsilon)

    agent.save(args.save_path)
    print(f"\nAgent saved to {args.save_path}")

    if args.evaluate > 0:
        evaluate_agent(agent, episodes=args.evaluate)

    print(f"\nFinal epsilon: {agent.epsilon:.3f}")
    print(f"Training steps: {agent.step_count}")


if __name__ == "__main__":
    main()