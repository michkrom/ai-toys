"""
Game controllers for Pong.

Implements:
- HumanController: Keyboard input
- AlgorithmicController: Simple AI with prediction
- PerfectController: Nearly perfect AI with variation
- NNController: Neural network agent
- QLearningController: Deep Q-Network agent
"""

import random
import time
from abc import ABC, abstractmethod


class BaseController(ABC):
    """Base controller class."""

    def __init__(self, side: str = "left"):
        self.side = side
        self.last_input = time.time()
        self._running = True
        self.terminal_initialized = False
        self.fd = None
        self.old_settings = None

    @abstractmethod
    def get_move(self, game) -> int:
        """Get the next move: -1 (up), 0 (stay), or 1 (down)."""
        pass

    def is_inactive(self) -> bool:
        """Check if controller has been inactive for more than 10 seconds."""
        return time.time() - self.last_input > 10

    def get_status(self) -> str:
        """Get controller status string."""
        return f"active (ε={getattr(self, 'epsilon', 0):.2f})" if hasattr(self, 'epsilon') else "active"

    def cleanup(self):
        """Clean up resources."""
        self._running = False


class HumanController(BaseController):
    """Human player controlled via keyboard input."""

    def __init__(self, side: str = "left"):
        super().__init__(side)
        self._move_buffer = 0

    def get_move(self, game) -> int:
        self.last_input = time.time()
        return self._move_buffer

    def set_move(self, move: int):
        """Set move from input handler."""
        self._move_buffer = move


class AlgorithmicController(BaseController):
    """Human-like AI with prediction, reaction delays, and errors."""

    def __init__(self, side: str = "left", difficulty: float = 0.95, sluggishness: int = 2):
        super().__init__(side)
        self.difficulty = difficulty

        # Sluggishness: paddle only moves every N frames (simulates a slow paddle).
        # Gating is per-frame so the paddle still moves at 1/N speed reliably.
        self.sluggishness = max(1, sluggishness)
        self._frame_index = 0

        # Reaction delay: frames to wait before responding (0-2 frames)
        self.reaction_delay = random.randint(0, 2)
        self._delay_counter = 0
        self._pending_move = 0

        # Attention: rarely lose track of the ball (2-4%)
        self.attention_span = random.uniform(0.96, 0.98)

        # Persistent per-rally tracking bias: makes the controller imperfect
        # but STABLE. No random noise is injected into the control each frame.
        self.base_prediction_error = 0.6
        self._pred_bias = random.uniform(-0.5, 0.5)

        # Momentum/overshoot tendency
        self._last_move = 0

        # Fatigue: performance degrades slightly over long rallies
        self._rally_length = 0
        self.max_rally_before_fatigue = 30

    def get_move(self, game) -> int:
        # Sluggishness gate: on skip frames the paddle holds still, so it
        # moves at 1/sluggishness speed but still tracks reliably.
        if self.sluggishness > 1:
            self._frame_index = (self._frame_index + 1) % self.sluggishness
            if self._frame_index != 0:
                return 0

        # Ball coming toward this paddle?
        ball_coming = (game.ball_vel[0] < 0) if self.side == "left" else (game.ball_vel[0] > 0)
        
        if not ball_coming:
            # Ball going away - reset rally counter, pick a fresh persistent
            # tracking bias for the next approach (consistent, not jittery)
            self._rally_length = 0
            self._pred_bias = random.uniform(-0.5, 0.5)
            if self.side == "left":
                paddle_y = game.paddle_left
            else:
                paddle_y = game.paddle_right
            paddle_center = paddle_y + game.paddle_height / 2
            target_center = game.height / 2
            if paddle_center < target_center - 1:
                return 1
            elif paddle_center > target_center + 1:
                return -1
            return 0

        self._rally_length += 1
        
        # Attention lapse: sometimes just don't react
        if random.random() > self.attention_span:
            # Lost track - just keep last command (no stall, no noise)
            return self._last_move

        # Reaction delay: queue move and execute after delay
        if self._delay_counter > 0:
            self._delay_counter -= 1
            if self._delay_counter == 0:
                # Execute pending move
                self._last_move = self._pending_move
                return self._last_move
            else:
                # Still waiting - continue last move or stay
                return self._last_move

        # Predict where ball will hit paddle
        ball_pos = game.ball_pos
        ball_vel = game.ball_vel

        if self.side == "left":
            target_x = 0
            paddle_y = game.paddle_left
        else:
            target_x = game.width
            paddle_y = game.paddle_right

        # Prediction time (frames until the ball reaches the paddle plane)
        prediction_time = (target_x - ball_pos[0]) / max(0.1, abs(ball_vel[0]))
        predicted_y = ball_pos[1] + ball_vel[1] * prediction_time

        # The closer the ball is, the better it is assessed -> movement gets
        # smaller and more precise as it approaches (error vanishes at impact).
        nearness = min(max(prediction_time, 0.0) / 2.0, 1.0)  # 0 at paddle, 1 far away
        speed_factor = min(abs(ball_vel[0]) + abs(ball_vel[1]), 3.0) / 3.0  # 0-1
        fatigue_factor = min(self._rally_length / self.max_rally_before_fatigue, 1.0)
        # Persistent bias scaled by proximity/speed/fatigue - NOT per-frame noise
        error_magnitude = self.base_prediction_error * nearness * (1.0 + speed_factor * 0.5 + fatigue_factor * 0.3)
        predicted_y += self._pred_bias * error_magnitude

        # Bounce prediction off walls
        while predicted_y < 0 or predicted_y > game.height:
            if predicted_y < 0:
                predicted_y = -predicted_y
            if predicted_y > game.height:
                predicted_y = 2 * game.height - predicted_y

        paddle_center = paddle_y + game.paddle_height / 2

        # Determine intended move (deadband tightens near the paddle for
        # smaller corrections, so movement visibly decreases on approach)
        if predicted_y < paddle_center - 1:
            intended_move = -1
        elif predicted_y > paddle_center + 1:
            intended_move = 1
        else:
            intended_move = 0

        # Hysteresis - reduce jitter: never flip direction on a coin flip.
        # Only reverse when the ball clearly crossed to the other side
        # (needs extra clearance), and avoid stop/start chatter while far off.
        if self._last_move != 0 and intended_move == -self._last_move:
            if intended_move == -1 and predicted_y < paddle_center - 3:
                intended_move = -1
            elif intended_move == 1 and predicted_y > paddle_center + 3:
                intended_move = 1
            else:
                intended_move = self._last_move  # hold direction
        elif intended_move == 0 and self._last_move != 0:
            # Ball is within the deadband: only stop when we are close enough
            # that our assessment is trustworthy; otherwise keep sweeping.
            if nearness < 0.25:
                intended_move = 0
            else:
                intended_move = self._last_move

        # Difficulty: occasionally make a random move instead (keeps it
        # beatable, but rare so it doesn't look noisy)
        if random.random() > self.difficulty:
            intended_move = random.choice([-1, 0, 1])

        # Set up reaction delay for next frame
        self._delay_counter = self.reaction_delay
        self._pending_move = intended_move
        
        # For this frame, return last move (simulating reaction time)
        return self._last_move


class PerfectController(BaseController):
    """Perfect controller - always moves to the ball's predicted intercept.

    It only ever loses a point if the opponent somehow shoots the ball for an
    angle it physically cannot reach in time; with normal play it never misses
    on its own side.
    """

    def __init__(self, side: str = "left"):
        super().__init__(side)

    def get_move(self, game) -> int:
        if self.side == "left":
            paddle_y = game.paddle_left
            paddle_center = paddle_y + game.paddle_height / 2
            ball_coming = game.ball_vel[0] < 0
            plane_x = 0.0
        else:
            paddle_y = game.paddle_right
            paddle_center = paddle_y + game.paddle_height / 2
            ball_coming = game.ball_vel[0] > 0
            plane_x = float(game.width)

        if not ball_coming:
            # Ball moving away - return toward center.
            target_center = game.height / 2
            if paddle_center < target_center - 0.5:
                return 1
            elif paddle_center > target_center + 0.5:
                return -1
            return 0

        # Predict where the ball will be when it reaches this paddle's plane,
        # accounting for wall bounces, then head straight there.
        target_y = self._predict_intercept_y(game, plane_x)

        if target_y < paddle_center - 0.5:
            return -1
        elif target_y > paddle_center + 0.5:
            return 1
        return 0

    @staticmethod
    def _predict_intercept_y(game, plane_x: float) -> float:
        bx, by = game.ball_pos
        vx, vy = game.ball_vel
        if vx == 0:
            return by
        t = (plane_x - bx) / vx
        y = by + vy * t
        # Fold on the vertical span to simulate top/bottom wall bounces.
        span = float(game.height)
        y %= (2 * span)
        if y > span:
            y = 2 * span - y
        return max(0.0, min(span, y))


class NNController(BaseController):
    """Neural network controller."""

    def __init__(self, side: str = "left"):
        super().__init__(side)
        self._network = None

    def get_move(self, game) -> int:
        # Placeholder for neural network logic
        # This would use a trained network to make decisions
        ball_coming = (game.ball_vel[0] > 0) if self.side == "right" else (game.ball_vel[0] < 0)

        if not ball_coming:
            return 0

        if self.side == "left":
            paddle_y = game.paddle_left
        else:
            paddle_y = game.paddle_right

        paddle_center = paddle_y + game.paddle_height / 2

        if game.ball_pos[1] < paddle_center:
            return -1
        elif game.ball_pos[1] > paddle_center:
            return 1
        return 0


class QLearningController(BaseController):
    """
    Self-learning AI controller using Deep Q-Network.

    This controller learns through self-play, improving over time
    to become a strong Pong opponent.
    """

    def __init__(self, side: str = "left", agent=None):
        super().__init__(side)
        self.agent = agent
        self._last_action = None

    def get_move(self, game) -> int:
        if self.agent is None:
            return random.choice([-1, 0, 1])

        # Convert game state to agent state
        action = self.agent.get_action(game)
        self._last_action = action
        return action

    def _get_state(self, game):
        """Convert game state to normalized state vector."""
        # Get my paddle info
        if self.side == "right":
            my_paddle_center = (game.paddle_right + game.paddle_height / 2) / game.height
        else:
            my_paddle_center = (game.paddle_left + game.paddle_height / 2) / game.height

        # Normalize ball position
        if self.side == "right":
            ball_x = 1.0 - game.ball_pos[0] / game.width
            ball_y = game.ball_pos[1] / game.height
        else:
            ball_x = game.ball_pos[0] / game.width
            ball_y = game.ball_pos[1] / game.height

        return [ball_x, ball_y, my_paddle_center, 0.5, 0.5, 0.5, 0.0]


CONTROLLERS = {
    "human": HumanController,
    "algorithmic": AlgorithmicController,
    "perfect": PerfectController,
    "nn": NNController,
    "qlearning": QLearningController,
}