"""
Training utilities for Q-Learning agents.

Includes:
- SelfPlayTrainer: Train agents against each other
- train_against_perfect: Train against a perfect controller
- train_quick: Quick training helper
- compute_step_reward / bounced_off_my_paddle: shared reward helpers

All trainers snapshot the physics state BEFORE calling physics.update() -
PongPhysics mutates in place, so without a snapshot the stored (state,
next_state) pair would alias the same object and carry no transition info.
"""

import copy
import math

from .agent import QLearningAgent, REWARD_BOUNCE, REWARD_WIN, REWARD_LOSS

# Hard cap on steps per episode so a rally can never run forever.
MAX_EPISODE_STEPS = 5000


def _snapshot(physics):
    """Deep-copy snapshot: safe against in-place mutation by update()."""
    return copy.deepcopy(physics)


def _score_changed(prev, curr) -> bool:
    """True if either player scored during the step (ball was reset)."""
    return (
        curr.left_score != prev.left_score
        or curr.right_score != prev.right_score
    )


def bounced_off_my_paddle(side: str, prev, curr) -> bool:
    """True if the ball bounced off my paddle during the step.

    Reliable detection: the ball's x-velocity flips sign only on a paddle
    hit (wall bounces flip vy, not vx; score resets are excluded by the
    score-change guard, since a new serve also flips the sign).
    """
    if _score_changed(prev, curr):
        return False
    if side == "right":
        return prev.ball_vel[0] > 0 and curr.ball_vel[0] < 0
    return prev.ball_vel[0] < 0 and curr.ball_vel[0] > 0


def compute_step_reward(side: str, prev, curr, action: int) -> float:
    """Shaped per-step reward.

    - +REWARD_BOUNCE for bouncing the ball off my own paddle
    - while the ball is inbound toward my side: dense reward for being under
      the ball (0..1) plus a small bonus for moving toward it
    - 0.0 while the ball is moving away (nothing I do matters yet)
    """
    if bounced_off_my_paddle(side, prev, curr):
        return REWARD_BOUNCE

    if side == "right":
        my_center = curr.paddle_right + curr.paddle_height / 2
        ball_coming = curr.ball_vel[0] > 0
    else:
        my_center = curr.paddle_left + curr.paddle_height / 2
        ball_coming = curr.ball_vel[0] < 0

    if not ball_coming:
        return 0.0

    distance = abs(curr.ball_pos[1] - my_center)
    position_reward = 1.0 - min(distance / curr.height, 1.0)

    movement_reward = 0.0
    if action != 0:
        if (curr.ball_pos[1] > my_center and action == 1) or (
            curr.ball_pos[1] < my_center and action == -1
        ):
            movement_reward = 0.5

    return position_reward + movement_reward


def _miss_closeness(side: str, curr) -> float:
    """How close the paddle on `side` was to the ball when it got past them.

    1.0 = the ball crossed essentially right where the paddle is; decays
    smoothly with distance (a Gaussian kernel in paddle-halves).
    """
    miss_y = curr.last_miss_y
    if miss_y is None:
        return 0.0
    center = (
        curr.paddle_right if side == "right" else curr.paddle_left
    ) + curr.paddle_height / 2
    distance = abs(miss_y - center)
    half = curr.paddle_height / 2
    return math.exp(-(distance / half) ** 2)


def point_reward(agent_side: str, prev, curr):
    """Terminal point reward, shaped by how close the loser's paddle was to
    the ball when it got past them.

    Returns (reward, done):
      done   = True iff a point was scored this step.
      reward =  +10 * (0.1 + 0.9 * (1 - opp_closeness))   if agent won
              =   -5 * (1 - my_closeness)                 if agent lost

    A flat -5/10 tells the Q-function nothing about *how* the point was
    won or lost; making the penalty proportional to miss distance gives it
    a smooth gradient right at the failure, so it learns to shave the miss
    margin (and on wins, to make the opponent miss wide).
    """
    if (curr.left_score == prev.left_score
            and curr.right_score == prev.right_score):
        return 0.0, False

    if agent_side == "right":
        won = curr.right_score > prev.right_score
    else:
        won = curr.left_score > prev.left_score

    if won:
        # The opponent missed on their own side.
        opp_side = "left" if agent_side == "right" else "right"
        closeness = _miss_closeness(opp_side, curr)
        return REWARD_WIN * (0.1 + 0.9 * (1.0 - closeness)), True

    closeness = _miss_closeness(agent_side, curr)
    return REWARD_LOSS * (1.0 - closeness), True


class SelfPlayTrainer:
    """Trains agents through self-play."""

    def __init__(
        self, agent_class, episodes: int = 1000, opponent_epsilon: float = 0.05
    ):
        self.agent_class = agent_class
        self.episodes = episodes
        self.opponent_epsilon = opponent_epsilon
        self.agent = None
        self.opponent = None

    def train(self, physics_class, render: bool = False):
        print(f"Starting self-play training for {self.episodes} episodes...")

        self.agent = QLearningAgent(side="right")
        self.opponent = QLearningAgent(
            side="left", epsilon=self.opponent_epsilon
        )

        wins = 0
        losses = 0

        for episode in range(self.episodes):
            physics = physics_class()
            done = False
            step_count = 0

            while not done and step_count < MAX_EPISODE_STEPS:
                agent_action = self.agent.get_action(physics)
                opponent_action = self.opponent.get_action(physics)

                prev_state = _snapshot(physics)

                # update(left_move, right_move): agent is right, opponent left
                physics.update(opponent_action, agent_action)

                # Terminal point reward shaped by miss distance, or dense
                # per-step reward while the rally continues.
                reward, done = point_reward("right", prev_state, physics)
                if done:
                    opponent_reward, _ = point_reward("left", prev_state, physics)
                    if reward > 0:
                        wins += 1
                    else:
                        losses += 1
                else:
                    reward = compute_step_reward(
                        "right", prev_state, physics, agent_action
                    )
                    opponent_reward = compute_step_reward(
                        "left", prev_state, physics, opponent_action
                    )

                # Store the terminal scoring transition for BOTH agents:
                # the point outcome is the strongest learning signal there is.
                self.agent.store_experience(
                    prev_state, agent_action, reward, physics, done
                )
                self.agent.train_step()

                self.opponent.store_experience(
                    prev_state, opponent_action, opponent_reward, physics, done
                )
                self.opponent.train_step()

                step_count += 1

            if (episode + 1) % 100 == 0:
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
                print(
                    f"Episode {episode + 1}: Win rate = {win_rate:.2%}, "
                    f"Epsilon = {self.agent.epsilon:.3f}"
                )

        print(f"Training complete! Win rate: {wins / (wins + losses):.2%}")
        return self.agent


def train_against_perfect(
    physics_class, episodes: int = 500, opponent_side: str = "left"
) -> QLearningAgent:
    """Train agent against a perfect controller."""
    from controllers import PerfectController

    print(f"Training against perfect controller for {episodes} episodes...")

    agent = QLearningAgent(side="right" if opponent_side == "left" else "left")
    opponent = PerfectController(side=opponent_side)

    wins = 0
    losses = 0

    for episode in range(episodes):
        physics = physics_class()
        done = False
        step_count = 0

        while not done and step_count < MAX_EPISODE_STEPS:
            if agent.side == "right":
                agent_action = agent.get_action(physics)
                opponent_action = opponent.get_move(physics)
            else:
                opponent_action = opponent.get_move(physics)
                agent_action = agent.get_action(physics)

            prev_state = _snapshot(physics)

            # update(left_move, right_move): map each action to its own paddle
            if agent.side == "right":
                physics.update(opponent_action, agent_action)
            else:
                physics.update(agent_action, opponent_action)

            if physics.right_score > prev_state.right_score:
                reward, done = point_reward(agent.side, prev_state, physics)
                wins += 1
            elif physics.left_score > prev_state.left_score:
                reward, done = point_reward(agent.side, prev_state, physics)
                losses += 1
            else:
                reward = compute_step_reward(
                    agent.side, prev_state, physics, agent_action
                )
                done = False

            agent.store_experience(
                prev_state, agent_action, reward, physics, done
            )
            agent.train_step()

            step_count += 1

        if (episode + 1) % 100 == 0:
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            print(
                f"Episode {episode + 1}: Win rate = {win_rate:.2%}, "
                f"Epsilon = {agent.epsilon:.3f}"
            )

    print(f"Training complete! Win rate: {wins / (wins + losses):.2%}")
    return agent


def train_quick(physics_class, episodes: int = 500) -> QLearningAgent:
    """Quick training helper using self-play."""
    trainer = SelfPlayTrainer(QLearningAgent, episodes=episodes)
    return trainer.train(physics_class)