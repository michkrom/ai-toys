import json

class LoggingRenderer:
    """
    Renderer that logs positions to output for ML controller training in headless mode.
    
    Emits one STATE record per frame only when something *meaningful* happens:
      - ball velocity vector changed (wall/paddle bounce -> direction change)
      - a point was scored
      - a paddle actually moved (preserves the action history for RL)
    This avoids spamming constant-velocity frames while still capturing the full
    action history the ML controllers need to learn from.
    """

    def __init__(self, frame=0, last_ball_vel=None, last_left_score=0, last_right_score=0):
        self.frame = frame
        self.last_ball_vel = last_ball_vel
        self.last_left_score = last_left_score
        self.last_right_score = last_right_score

    def render(self, state, l_move=None, r_move=None, left_controller=None, right_controller=None):
        """Render state by logging meaningful events."""
        self.frame += 1

        # Handle initial None case
        if self.last_ball_vel is None:
            ball_changed = True
        else:
            ball_changed = self.last_ball_vel != list(state.ball_vel)
        score_changed = state.left_score != self.last_left_score or state.right_score != self.last_right_score
        action_taken = l_move not in (0, None) or r_move not in (0, None)

        if ball_changed or score_changed or action_taken:
            output = {
                'frame': self.frame,
                'ball': {
                    'x': state.ball_pos[0],
                    'y': state.ball_pos[1]
                },
                'vel_x': state.ball_vel[0],
                'vel_y': state.ball_vel[1],
                'paddles': {
                    'left': state.paddle_left,
                    'right': state.paddle_right
                },
                'height': state.paddle_height,
                'scores': {
                    'left': state.left_score,
                    'right': state.right_score
                },
                'actions': {
                    'left': l_move,
                    'right': r_move
                }
            }
            print(f"STATE: {json.dumps(output)}")

            self.last_ball_vel = list(state.ball_vel)
            self.last_left_score = state.left_score
            self.last_right_score = state.right_score

    def get_status(self):
        return "Logging"