"""
Training script for Q-Learning Pong agent.

This script trains a Q-learning agent to play Pong using:
- Deep Q-Network with experience replay
- Epsilon-greedy exploration
- Custom reward function that encourages ball bouncing

Usage:
    python train_qlearning.py [--episodes N] [--epsilon E]

Examples:
    # Train for 100 episodes with default epsilon
    python train_qlearning.py

    # Train for 200 episodes with high exploration
    python train_qlearning.py --episodes 200 --epsilon 0.9

    # Quick test with 50 episodes
    python train_qlearning.py --episodes 50 --epsilon 0.8
"""

import argparse
import random
import sys

from qlearning import QLearningAgent
from qlearning.trainer import (
    compute_step_reward,
    bounced_off_my_paddle,
    point_reward,
    _snapshot,
    MAX_EPISODE_STEPS,
)


def _make_opponent(name: str):
    """Return (get_move, update_call) helpers for the chosen opponent.

    The agent always plays the right side; opponents act on the left.
    update_call(physics, opponent_move, agent_action) applies both moves.
    """
    from physics import PongPhysics

    if name == "stationary":
        return None, lambda p, om, am: p.update(0, am)

    from controllers import AlgorithmicController, PerfectController
    cls = {"perfect": PerfectController, "algorithmic": AlgorithmicController}[name]
    opp = cls(side="left")
    return opp.get_move, lambda p, om, am: p.update(om, am)


def _seed_rng(seed):
    """Seed both the stdlib and numpy RNGs (networks are numpy-backed)."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_agent(episodes=100, epsilon=0.8, verbose=True,
                interval=25, opponent="stationary"):
    """
    Train the Q-learning agent (right side).

    Args:
        episodes: Number of training episodes
        epsilon: Initial epsilon (exploration rate)
        verbose: Whether to print progress
        interval: Report metrics every N episodes (0 disables periodic output)
        opponent: "stationary", "algorithmic" or "perfect" (left side)
    """
    from physics import PongPhysics

    agent = QLearningAgent(side="right", epsilon=epsilon)
    opp_get_move, do_update = _make_opponent(opponent)

    total_bounces = 0
    total_wins = 0
    total_losses = 0
    total_steps = 0

    # Per-interval counters for the monitoring view: bounces vs points lost
    int_bounces = 0
    int_wins = 0
    int_losses = 0
    int_steps = 0

    print(f"Training agent for {episodes} episodes vs {opponent} opponent")
    print("=" * 86)
    print(f"{'ep':>5} | {'bounces':>7} {'/ep':>5} | {'W':>3} {'L':>3} "
          f"{'W%':>5} | {'rally':>5} | {'eps':>5}")
    print("-" * 86)

    for episode in range(1, episodes + 1):
        physics = PongPhysics()
        step_count = 0

        while step_count < MAX_EPISODE_STEPS:
            agent_action = agent.get_action(physics)

            # Snapshot BEFORE update: physics mutates in place, so the stored
            # (state, next_state) must come from distinct points in time.
            prev_state = _snapshot(physics)

            opp_move = opp_get_move(physics) if opp_get_move else 0
            do_update(physics, opp_move, agent_action)

            if bounced_off_my_paddle("right", prev_state, physics):
                total_bounces += 1
                int_bounces += 1

            reward, done = point_reward("right", prev_state, physics)
            if not done:
                reward = compute_step_reward(
                    "right", prev_state, physics, agent_action
                )
            else:
                if reward > 0:
                    total_wins += 1
                    int_wins += 1
                else:
                    total_losses += 1
                    int_losses += 1

            agent.store_experience(
                prev_state, agent_action, reward, physics, done
            )
            agent.train_step()

            step_count += 1
            total_steps += 1
            int_steps += 1
            if done:
                break

        if interval and verbose and episode % interval == 0:
            denom = int_wins + int_losses
            win_pct = 100.0 * int_wins / denom if denom else float("nan")
            rally = int_steps / interval if interval else 0.0
            print(
                f"{episode:5d} | {int_bounces:7d} {int_bounces/interval:5.1f} | "
                f"{int_wins:3d} {int_losses:3d} {win_pct:5.1f} | "
                f"{rally:5.1f} | {agent.epsilon:5.3f}"
            )
            int_bounces = 0
            int_wins = 0
            int_losses = 0
            int_steps = 0

    print("=" * 86)
    print("Training complete!")
    print(f"  Total bounces: {total_bounces} "
          f"({total_bounces / episodes:.2f}/episode)")
    print(f"  Wins: {total_wins}, Losses: {total_losses}")
    if total_wins + total_losses > 0:
        print(f"  Win rate: {total_wins / (total_wins + total_losses):.2%}")
    print(f"  Steps: {total_steps} ({total_steps / episodes:.1f}/episode)")

    return agent


def test_agent(agent, steps=20):
    """
    Test the trained agent.

    Args:
        agent: Trained agent
        steps: Number of steps to run
    """
    from physics import PongPhysics

    print("\n" + "=" * 60)
    print("Testing trained agent...")
    print("=" * 60)

    physics = PongPhysics()

    for step in range(steps):
        action = agent.get_action(physics)
        state = agent._get_state(physics)
        q_values = agent.q_network.forward(state)[0]

        my_paddle_center = (
            physics.paddle_right + physics.paddle_height / 2
        )

        print(
            f"Step {step:2d}: Action={action:2d} | "
            f"Q={[round(q, 2) for q in q_values]} | "
            f"Paddle={my_paddle_center:.1f} | Ball={physics.ball_pos}"
        )

        prev_left_score = physics.left_score
        prev_right_score = physics.right_score

        physics.update(0, action)

        if physics.right_score > prev_right_score:
            print(f"  -> SCORED! (Score: {physics.right_score})")
            return

        if physics.left_score > prev_left_score:
            print(f"  -> LOST (Score: {physics.left_score})")
            return

    print("\nDone testing")


def main():
    parser = argparse.ArgumentParser(
        description='Train Q-Learning Pong Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_qlearning.py                          # Train 100 episodes vs stationary
  python train_qlearning.py --episodes 300 --interval 25 --opponent algorithmic
  python train_qlearning.py --epsilon 0.9            # Higher exploration
        """,
    )
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes (default: 100)')
    parser.add_argument('--epsilon', type=float, default=0.8,
                        help='Initial exploration rate 0-1 (default: 0.8)')
    parser.add_argument('--interval', type=int, default=25,
                        help='Report bounces/W/L every N episodes (default: 25)')
    parser.add_argument('--opponent', type=str, default='stationary',
                        choices=['stationary', 'algorithmic', 'perfect'],
                        help='Left-side opponent (default: stationary)')
    parser.add_argument('--save', type=str, default='',
                        help='Save the trained agent to this path '
                             '(default: don\'t save)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible runs '
                             '(default: random)')
    parser.add_argument('--test-steps', type=int, default=20,
                        help='Number of test steps (default: 20)')

    args = parser.parse_args()

    if args.seed is not None:
        _seed_rng(args.seed)

    agent = train_agent(
        episodes=args.episodes,
        epsilon=args.epsilon,
        interval=args.interval,
        opponent=args.opponent,
    )
    if args.save:
        agent.save(args.save)
        print(f"Saved agent to {args.save}")
    test_agent(agent, steps=args.test_steps)


if __name__ == "__main__":
    sys.exit(main())