"""
Renderers for Pong game.

Implements:
- BaseRenderer: Abstract base class
- TUIRenderer: Terminal-based renderer using blessed
- LoggingRenderer: Logs state for ML training
- SDL2Renderer: Pygame-based renderer
"""

from blessed import Terminal
import sys
import json


BIG_NUMBERS = {
    '0': [' ███ ', '█   █', '█   █', '█   █', ' ███ '],
    '1': ['  █  ', ' ██  ', '  █  ', '  █  ', ' ████'],
    '2': [' ███ ', '█   █', '   █ ', '  █  ', '█████'],
    '3': [' ███ ', '    █', '  ██ ', '    █', ' ███ '],
    '4': ['   █ ', '  ██ ', ' █ █ ', '█████', '   █ '],
    '5': ['█████', '█    ', ' ███ ', '    █', ' ███ '],
    '6': ['  █  ', ' █   ', '████ ', '█   █', ' ███ '],
    '7': ['█████', '    █', '   █ ', '  █  ', '  █  '],
    '8': [' ███ ', '█   █', ' ███ ', '█   █', ' ███ '],
    '9': [' ███ ', '█   █', ' ████', '    █', ' ███ '],
}


def draw_large_number(frame, num_str, start_col, height):
    """Draw a large number using 5x3 block patterns."""
    for digit_idx, digit in enumerate(num_str):
        if digit not in BIG_NUMBERS:
            continue
        pattern = BIG_NUMBERS[digit]
        for row_idx, row in enumerate(pattern):
            if row_idx >= height - 1:
                continue
            for col_idx, char in enumerate(row):
                col_pos = start_col + col_idx + digit_idx * 6
                if col_pos < len(frame[0]):
                    if char == '█':
                        frame[row_idx + 1][col_pos] = '█'


class BaseRenderer:
    """Base renderer class."""

    def render(self, state):
        raise NotImplementedError("Subclasses must implement render()")


class TUIRenderer(BaseRenderer):
    """Terminal-based renderer using blessed."""

    def __init__(self):
        self.term = Terminal()

    def render(self, state, left_controller=None, right_controller=None, l_move=None, r_move=None):
        frame_height = state.height          # rows in the field (0..height-1)
        last_row = state.height - 1          # highest row index we draw into
        frame = []
        for _ in range(state.height):
            frame.append([' '] * state.width)

        mid = state.width // 2

        # Draw scores
        left_score = str(state.left_score)
        right_score = str(state.right_score)
        draw_large_number(frame, left_score, 30, frame_height)
        draw_large_number(frame, right_score, mid + 2, frame_height)

        # Draw paddles (aligned with the physics collision planes: left at
        # column 0 / x=0.5, right at the last column / x=width-0.5)
        for i in range(state.paddle_height):
            y_pos = max(0, min(last_row, int(state.paddle_left) + i))
            if 0 <= y_pos < len(frame) and 0 < state.width:
                frame[y_pos][0] = '█'

        for i in range(state.paddle_height):
            y_pos = max(0, min(last_row, int(state.paddle_right) + i))
            if 0 <= y_pos < len(frame) and 2 < state.width:
                frame[y_pos][state.width - 1] = '█'

        # Draw ball
        ball_x = max(0, min(state.width, int(state.ball_pos[0])))
        ball_y = max(0, min(last_row, int(state.ball_pos[1])))
        if 0 <= ball_y < len(frame) and 0 <= ball_x < state.width:
            frame[ball_y][ball_x] = '●'

        # Clear screen and print every row (row height-1 is a valid ball row)
        print(self.term.clear(), end='')

        for y in range(frame_height):
            print(''.join(frame[y]))

        # Print status on the line below the field
        left_name = left_controller.__class__.__name__ if left_controller else 'None'
        right_name = right_controller.__class__.__name__ if right_controller else 'None'
        left_status = left_controller.get_status() if left_controller and hasattr(left_controller, 'get_status') else 'Unknown'
        right_status = right_controller.get_status() if right_controller and hasattr(right_controller, 'get_status') else 'Unknown'

        status_text = f" [L: {left_name} ({left_status})]  [R: {right_name} ({right_status})] "
        print(self.term.move_xy(0, frame_height), status_text)
        sys.stdout.flush()


class LoggingRenderer(BaseRenderer):
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

    def render(self, state, left_controller=None, right_controller=None, l_move=None, r_move=None):
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


class SDL2Renderer(BaseRenderer):
    """SDL2-based renderer using pygame."""

    def __init__(self):
        try:
            import pygame
            self.pygame = pygame
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Pong Game')
            self.clock = pygame.time.Clock()
        except ImportError:
            self.screen = None
            print('Pygame not installed. Install with: pip install pygame')

    def render(self, state, left_controller=None, right_controller=None, l_move=None, r_move=None):
        if not self.screen:
            return

        left_status = left_controller.get_status() if left_controller and hasattr(left_controller, 'get_status') else 'Unknown'
        right_status = right_controller.get_status() if right_controller and hasattr(right_controller, 'get_status') else 'Unknown'
        status_text = f"L: {left_status} | R: {right_status}"
        self.pygame.display.set_caption(f"Pong - Score: {state.left_score} - {state.right_score} | {status_text}")

        self.clock.tick(60)
        self.screen.fill((0, 0, 0))

        left_paddle = self.pygame.Rect(0, int(state.paddle_left) * 10, 10, 50)
        right_paddle = self.pygame.Rect((state.width - 1) * 10, int(state.paddle_right) * 10, 10, 50)

        self.pygame.draw.rect(self.screen, (255, 255, 255), left_paddle)
        self.pygame.draw.rect(self.screen, (255, 255, 255), right_paddle)

        ball_rect = self.pygame.Rect(int(state.ball_pos[0]) * 10, int(state.ball_pos[1]) * 10, 10, 10)
        self.pygame.draw.ellipse(self.screen, (255, 255, 255), ball_rect)
        self.pygame.display.flip()

    def cleanup(self):
        if hasattr(self, 'pygame') and self.screen:
            self.pygame.quit()
            return None
        return None

    def get_status(self):
        return "SDL2"