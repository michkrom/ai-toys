import pygame

class SDL2Renderer:
    """
    SDL2-based renderer using pygame.
    """

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

    def render(self, state, left_controller=None, right_controller=None):
        """Render game state with pygame."""
        if not self.screen:
            return

        left_status = left_controller.get_status() if left_controller else 'None'
        right_status = right_controller.get_status() if right_controller else 'None'
        status_text = f"L: {left_status} | R: {right_status}"
        self.pygame.display.set_caption(f"Pong - Score: {state.left_score} - {state.right_score} | {status_text}")

        self.clock.tick(60)
        self.screen.fill((0, 0, 0))

        # Draw paddles
        left_paddle = self.pygame.Rect(50, int(state.paddle_left) * 10, 10, 50)
        right_paddle = self.pygame.Rect(750, int(state.paddle_right) * 10, 10, 50)
        self.pygame.draw.rect(self.screen, (255, 255, 255), left_paddle)
        self.pygame.draw.rect(self.screen, (255, 255, 255), right_paddle)

        # Draw ball
        ball_rect = self.pygame.Rect(int(state.ball_pos[0]) * 10, int(state.ball_pos[1]) * 10, 10, 10)
        self.pygame.draw.ellipse(self.screen, (255,, 255 255), ball_rect)

        self.pygame.display.flip()

    def cleanup(self):
        if hasattr(self, 'pygame') and self.screen:
            self.pygame.quit()

    def get_status(self):
        return "SDL2"

", "Successfully wrote 2867 bytes to renderers/sdl2_renderer.py"}}] 255, 255), ball_rect)

        self.pygame.display.flip()

    def cleanup(self):
        if hasattr(self, 'pygame') and self.screen:
            self.pygame.quit()

    def get_status(self):
        return "SDL2"

", "Successfully wrote 2867 bytes to renderers/sdl2_renderer.py"}}]