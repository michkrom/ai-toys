"""
Game physics engine for Pong.

Implements classic Pong mechanics with proper collision detection order:
- Wall bounces (top/bottom)
- Paddle collisions (swept detection with y-validation)
- Scoring (ball passed paddle without collision)
"""

import random
import math


class PongPhysics:
    """Classic Pong physics engine with predictive collision handling."""

    MAX_VERTICAL_VELOCITY = 0.7
    MIN_SPEED = 0.3
    MAX_SPEED = 1.5

    def __init__(self, width: int = 80, height: int = 25):
        self.width = width
        self.height = height
        # Initialize ball at center
        self.ball_pos = [float(width) / 2, float(height) / 2]
        # Initial diagonal movement
        self.ball_vel = [0.5, 0.5]
        
        # Initialize paddles - both centered vertically
        self.paddle_left = float(height) / 2
        self.paddle_right = float(height) / 2
        self.paddle_height = 5
        self.paddle_width = 1
        
        # Score tracking
        self.left_score = 0
        self.right_score = 0
        self.game_over = False
        self._last_hit_frame = 0
        self._frame_count = 0
        self._server_side = 1

        # Where the ball got past a paddle on the most recent point:
        # last_miss_side is the player whose paddle failed ("left"/"right"),
        # last_miss_y is the ball's y at the moment it crossed the wall.
        # Used to shape the point-lost reward by miss distance.
        self.last_miss_side = None
        self.last_miss_y = None

    def update(self, left_move: int, right_move: int):
        """Update game state with proper collision handling order."""
        self._frame_count += 1

        # Store snapshot of current state
        prev_x = self.ball_pos[0]
        prev_y = self.ball_pos[1]
        prev_vx = self.ball_vel[0]
        prev_vy = self.ball_vel[1]

        # Update paddle positions (with clamping)
        self.paddle_left = max(0, min(self.height - self.paddle_height, self.paddle_left + left_move))
        self.paddle_right = max(0, min(self.height - self.paddle_height, self.paddle_right + right_move))

        # Predict future position
        future_x = prev_x + prev_vx
        future_y = prev_y + prev_vy

        # ====== PRIORITY 1: WALL BOUNCES ======
        # Wall boundaries match the position clamp used below, so even a tiny
        # vertical velocity always escapes the wall instead of hovering on it.
        MIN_Y = 0.5
        MAX_Y = self.height - 0.5

        # Top wall - reflect position around MIN_Y and invert vertical velocity
        if future_y < MIN_Y:
            future_y = 2 * MIN_Y - future_y
            self.ball_vel[1] = -self.ball_vel[1]
        # Bottom wall - reflect position around MAX_Y and invert vertical velocity
        elif future_y > MAX_Y:
            future_y = 2 * MAX_Y - future_y
            self.ball_vel[1] = -self.ball_vel[1]

        # ====== PRIORITY 2: PADDLE COLLISIONS ======
        handled_collision = False
        paddle_zone_x = 0.5  # Left paddle zone boundary
        paddle_zone_right_x = self.width - 0.5  # Right paddle zone boundary

        # Left paddle collision
        if prev_vx < 0 and prev_x >= paddle_zone_x and future_x < paddle_zone_x:
            # Ball moving left and crossing into paddle zone
            if prev_x != future_x:
                t = (paddle_zone_x - prev_x) / (future_x - prev_x)
                collision_y = prev_y + t * (future_y - prev_y)
            else:
                t = 0.0
                collision_y = future_y

            # Check if collision is within paddle's vertical range
            paddle_top = self.paddle_left
            paddle_bottom = self.paddle_left + self.paddle_height
            if paddle_top <= collision_y <= paddle_bottom:
                self._handle_paddle_collision("left", left_move, collision_y)
                handled_collision = True
                # Bounce: advance the rest of the frame with the new velocity
                # so the ball is drawn *facing away* from the paddle, not behind it.
                remaining = 1.0 - t
                future_x = paddle_zone_x + self.ball_vel[0] * remaining
                future_y = collision_y + self.ball_vel[1] * remaining

        # Right paddle collision
        if not handled_collision and prev_vx > 0 and prev_x < paddle_zone_right_x and future_x >= paddle_zone_right_x:
            if prev_x != future_x:
                t = (paddle_zone_right_x - prev_x) / (future_x - prev_x)
                collision_y = prev_y + t * (future_y - prev_y)
            else:
                t = 0.0
                collision_y = prev_y

            paddle_top = self.paddle_right
            paddle_bottom = self.paddle_right + self.paddle_height
            if paddle_top <= collision_y <= paddle_bottom:
                self._handle_paddle_collision("right", right_move, collision_y)
                handled_collision = True
                # Bounce: advance the rest of the frame with the new velocity
                # so the ball is drawn *facing away* from the paddle, not behind it.
                remaining = 1.0 - t
                future_x = paddle_zone_right_x + self.ball_vel[0] * remaining
                future_y = collision_y + self.ball_vel[1] * remaining

        # ====== PRIORITY 3: SCORING ======
        # Ball passed left wall (missed left paddle) -> right player scores.
        # Serve toward the opponent (right side).
        if not handled_collision and future_x <= 0:
            self.right_score = (self.right_score + 1) % 100
            self.last_miss_side = "left"
            self.last_miss_y = future_y
            self._reset_ball(1)
            return

        # Ball passed right wall (missed right paddle) -> left player scores.
        # Serve toward the opponent (left side).
        if not handled_collision and future_x >= self.width:
            self.left_score = (self.left_score + 1) % 100
            self.last_miss_side = "right"
            self.last_miss_y = future_y
            self._reset_ball(-1)
            return

        # ====== FINAL POSITION UPDATE ======
        # Clamp position
        future_x = max(0.5, min(self.width - 0.5, future_x))
        future_y = max(0.5, min(self.height - 0.5, future_y))

        # Keep ball speed bounded (min and max) so long rallies stay stable
        speed = math.sqrt(self.ball_vel[0]**2 + self.ball_vel[1]**2)
        if speed < self.MIN_SPEED:
            scale = self.MIN_SPEED / speed
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale
        elif speed > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed
            self.ball_vel[0] *= scale
            self.ball_vel[1] *= scale

        self.ball_pos = [future_x, future_y]

    def _handle_paddle_collision(self, side: str, paddle_move: int, collision_y: float):
        """Apply collision physics when ball hits paddle."""
        # Calculate hit offset relative to paddle center
        if side == "left":
            paddle_center = self.paddle_left + self.paddle_height / 2
        else:
            paddle_center = self.paddle_right + self.paddle_height / 2

        hit_offset = (collision_y - paddle_center) / (self.paddle_height / 2)
        hit_offset = max(-1.0, min(1.0, hit_offset))

        # Calculate angle factor based on hit zone
        angle_factor = self._get_zone_angle(hit_offset)

        # Calculate new velocity
        speed_mult = 1.15 + 0.1 * abs(angle_factor)
        new_vx = -self.ball_vel[0] * speed_mult
        new_vy = angle_factor * self.MAX_VERTICAL_VELOCITY + paddle_move * 0.12

        # Ensure non-zero vertical velocity
        if abs(new_vy) < 0.05:
            new_vy = 0.05 if self.ball_vel[1] > 0 else -0.05

        # Clamp vertical velocity
        new_vy = max(-self.MAX_VERTICAL_VELOCITY, min(self.MAX_VERTICAL_VELOCITY, new_vy))

        # Apply new velocity
        self.ball_vel = [new_vx, new_vy]
        self._last_hit_frame = self._frame_count

    def _get_zone_angle(self, hit_offset: float) -> float:
        """Calculate bounce angle factor based on where the ball hits the paddle.

        hit_offset ranges from -1 (top) to +1 (bottom) relative to paddle center.
        - Top zone (offset < -0.33): steep upward bounce (angle factor > 0)
        - Middle zone (|offset| < 0.33): straight/flat bounce (~0)
        - Bottom zone (offset > 0.33): steep downward bounce (angle factor < 0)
        """
        hit_offset = max(-1.0, min(1.0, hit_offset))
        abs_offset = abs(hit_offset)

        if abs_offset < 0.33:
            # Middle zone - gentle angle, proportional to offset
            return hit_offset * 1.5
        elif hit_offset > 0:
            # Bottom zone - steep downward bounce
            normalized = (hit_offset - 0.33) / 0.67  # 0 to 1 within zone
            return -(0.3 + normalized * 0.7)  # -0.3 to -1.0
        else:
            # Top zone - steep upward bounce
            normalized = (abs_offset - 0.33) / 0.67  # 0 to 1 within zone
            return (0.3 + normalized * 0.7)  # +0.3 to +1.0

    def _get_paddle_zone_name(self, hit_offset: float) -> str:
        """Return the name of the zone where the ball hit the paddle."""
        abs_offset = abs(hit_offset)
        if abs_offset < 0.33:
            return "middle"
        elif hit_offset > 0:
            return "bottom"
        else:
            return "top"

    def _reset_ball(self, direction: int = 1):
        """Reset ball to center for a new serve."""
        self.ball_pos = [float(self.width) / 2, float(self.height) / 2]
        angle = random.choice([-0.3, -0.15, 0.15, 0.3])  # Never zero to ensure vertical movement
        self.ball_vel = [0.5 * direction, 0.5 * angle]
        self._last_hit_frame = self._frame_count
        self._server_side = direction

    def game_over(self):
        """Check if game is over."""
        return False