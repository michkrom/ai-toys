"""
Q-Learning agent with Deep Q-Network.

Implements:
- Epsilon-greedy action selection with Double DQN
- Experience replay with prioritized sampling and IS-weighted updates
- Pure, stateless state featurization (a clean function of the physics
  object; no hidden history that could desync action-time from store-time)
"""

import random
from typing import List, Optional

from .network import QNetwork
from .replay_buffer import ReplayBuffer

# Action encoding: -1 (up) -> 0, 0 (stay) -> 1, 1 (down) -> 2
ACTION_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_ACTION = {0: -1, 1: 0, 2: 1}

# Rewards shared by the agent and the trainers
REWARD_BOUNCE = 5.0
REWARD_WIN = 10.0
REWARD_LOSS = -5.0


class QLearningAgent:
    """
    Self-learning AI controller using Deep Q-Network.
    """

    def __init__(
        self,
        side: str = "right",
        learning_rate: float = 0.001,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.02,
        gamma: float = 0.95,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        prioritized: bool = True,  # Enable prioritized replay
        priority_alpha: float = 0.6,  # Prioritization exponent
        priority_beta: float = 0.4,   # Importance sampling exponent
        priority_eps: float = 1e-6,   # Small constant for numerical stability
        hidden_size: int = 64,        # Hidden layer width of the Q-networks
    ):
        self.side = side
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.prioritized = prioritized
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        self.priority_eps = priority_eps
        self.hidden_size = hidden_size

        # Neural networks
        self.q_network = QNetwork(
            input_size=7, hidden_size=hidden_size, output_size=3
        )
        self.target_network = QNetwork(
            input_size=7, hidden_size=hidden_size, output_size=3)
        self.target_network.copy_weights(self.q_network)

        # Experience replay
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            prioritized=prioritized,
            alpha=priority_alpha,
            beta=priority_beta,
            eps=priority_eps,
        )

        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.last_action = None

        # Performance tracking
        self.win_count = 0
        self.loss_count = 0

        print(f"Double DQN agent initialized with epsilon={epsilon:.3f}, "
              f"prioritized={prioritized}")

    # ------------------------------------------------------------------
    # State representation
    # ------------------------------------------------------------------
    def _get_state(self, physics) -> List[float]:
        """
        Convert physics state to a normalized 7-D input vector.

        Pure function of the physics object: the same physics state always
        yields the same vector, so the vector stored in the replay buffer
        is guaranteed identical to the one the action was selected from.

        Features (all from the controller's point of view, in [~0, 1]):
            0. ball_x          (1.0 at my paddle, 0.0 at the far wall)
            1. ball_y
            2. my_paddle_center
            3. opponent_paddle_center
            4. ball_vx         (signed, +1.0 = moving toward my paddle)
            5. ball_vy         (signed, +1.0 = moving down)
            6. paddle_offset   (ball_y - my_paddle_center)

        Velocity comes straight from physics.ball_vel (per-frame steps make
        displacement == velocity in normal flight, but displacement is
        corrupted exactly at bounce and score-reset frames, so the physics
        value is used instead of a position-history proxy).
        """
        if self.side == "right":
            my_paddle_center = (
                physics.paddle_right + physics.paddle_height / 2
            ) / physics.height
            opp_paddle_center = (
                physics.paddle_left + physics.paddle_height / 2
            ) / physics.height
            # POV mirror: x = 1.0 at my paddle, velocity toward me is +vx
            ball_x = 1.0 - physics.ball_pos[0] / physics.width
            vx = -physics.ball_vel[0] / physics.MAX_SPEED
        else:
            my_paddle_center = (
                physics.paddle_left + physics.paddle_height / 2
            ) / physics.height
            opp_paddle_center = (
                physics.paddle_right + physics.paddle_height / 2
            ) / physics.height
            ball_x = physics.ball_pos[0] / physics.width
            vx = physics.ball_vel[0] / physics.MAX_SPEED

        ball_y = physics.ball_pos[1] / physics.height
        vy = physics.ball_vel[1] / physics.MAX_VERTICAL_VELOCITY
        paddle_offset = ball_y - my_paddle_center

        return [
            ball_x,
            ball_y,
            my_paddle_center,
            opp_paddle_center,
            vx,
            vy,
            paddle_offset,
        ]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def get_action(self, state) -> int:
        """Select an action via epsilon-greedy. Returns -1 (up), 0 (stay),
        or 1 (down)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            action = random.choice([-1, 0, 1])
            self.last_action = action
            return action

        state_vector = self._get_state(state)
        q_values, _, _ = self.q_network.forward(state_vector)
        action = INDEX_TO_ACTION[q_values.index(max(q_values))]
        self.last_action = action
        return action

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------
    def store_experience(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        info: dict = None,
    ):
        """Store experience in replay buffer with a real TD-error priority."""
        state_vector = self._get_state(state)
        next_state_vector = self._get_state(next_state)
        action_idx = ACTION_TO_INDEX[action]

        priority = self._td_error(
            state_vector, action_idx, reward, next_state_vector, done
        )
        self.replay_buffer.push(
            state_vector,
            action_idx,
            reward,
            next_state_vector,
            done,
            info or {},
            priority=priority,
        )

    def _td_error(
        self,
        state_vector: List[float],
        action_idx: int,
        reward: float,
        next_state_vector: List[float],
        done: bool,
    ) -> float:
        """Compute |TD error| for priority: |r + gamma*Q'(s', a') - Q(s, a)|.

        Uses the Double DQN convention: the online network picks the next
        action a', the target network evaluates it.
        """
        current_q, _, _ = self.q_network.forward(state_vector)
        q_current = current_q[action_idx]

        if done:
            target = reward
        else:
            online_next, _, _ = self.q_network.forward(next_state_vector)
            next_action = online_next.index(max(online_next))
            target_next, _, _ = self.target_network.forward(next_state_vector)
            target = reward + self.gamma * target_next[next_action]

        return abs(target - q_current)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def train_step(self) -> float:
        """One training step over a replayed batch.

        The stored next_state vectors are used directly - they were already
        computed at store time and must not be re-derived from live objects.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        total_loss = 0.0
        new_priorities = []

        for weight, experience in zip(weights, batch):
            state, action, reward, next_state, done, info = experience
            q_values, _, _ = self.q_network.forward(state)
            current_q = q_values[action]

            if done:
                target_q = reward
            else:
                # Double DQN: online net selects a', target net evaluates it
                online_next, _, _ = self.q_network.forward(next_state)
                next_action = online_next.index(max(online_next))
                target_next, _, _ = self.target_network.forward(next_state)
                target_q = reward + self.gamma * target_next[next_action]

            # IS-weighted MSE update for this sample
            targets = [target_q if i == action else q_values[i]
                       for i in range(3)]
            self.q_network.backward(
                state, targets, self.learning_rate, weight
            )

            error = abs(target_q - current_q)
            total_loss += error
            new_priorities.append(error + self.priority_eps)

        self.replay_buffer.update_priorities(indices, new_priorities)

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.copy_weights(self.q_network)

        return total_loss / len(batch)

    def update_from_game_event(self, event_type: str):
        """Track score events (wins/losses as counts, not reward magnitudes)."""
        if event_type == "point_win":
            self.win_count += 1
        elif event_type == "point_loss":
            self.loss_count += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: str):
        """Save network weights to file."""
        self.q_network.save(filepath)

    def load(self, filepath: str):
        """Load network weights from file."""
        self.q_network.load(filepath)
        self.target_network.copy_weights(self.q_network)