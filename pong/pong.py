#!/home/m/.venv/bin/python3
"""
Pong Game - Terminal Pong with AI Controllers.

A simple, cross-platform terminal-based Pong game with multiple AI controllers.

Controls:
    W or ↑ - Move left paddle up
    S or ↓ - Move left paddle down
    Q - Quit game
"""

import argparse
import os
import time
from controllers import (
    CONTROLLERS,
    HumanController,
    AlgorithmicController,
    QLearningController,
)
from physics import PongPhysics
from renderers import TUIRenderer, SDL2Renderer, LoggingRenderer


def _load_agent(side: str, path: str):
    """Load a trained Q-learning agent for a game controller.

    Returns None (random play) if no saved agent exists, with a warning.
    """
    if not os.path.exists(path):
        print(f"[pong] No saved agent at '{path}' - qlearning controller for "
              f"'{side}' side will play randomly. Train one first:"
              f"\n       invoke train.qlearning --episodes 300")
        return None
    from qlearning import QLearningAgent
    agent = QLearningAgent(side=side)
    agent.load(path)
    agent.epsilon = 0.0  # play greedily in the live game
    print(f"[pong] Loaded qlearning agent '{path}' for '{side}' side")
    return agent


class PongGame:
    """Main Pong game class."""

    def __init__(
        self,
        left_controller: str = "perfect",
        right_controller: str = "algorithmic",
        renderer: str = "TUI",
        agent_path: str = "trained_agent.pkl",
    ):
        self.physics = PongPhysics()

        # Get controller classes
        left_cls = CONTROLLERS.get(left_controller, AlgorithmicController)
        right_cls = CONTROLLERS.get(right_controller, AlgorithmicController)

        # Create controllers
        if left_cls == HumanController:
            self.left_ctrl = HumanController()
        elif left_cls == QLearningController:
            self.left_ctrl = QLearningController(
                side="left", agent=_load_agent("left", agent_path)
            )
        else:
            self.left_ctrl = left_cls(side="left")

        if right_cls == HumanController:
            self.right_ctrl = HumanController()
        elif right_cls == QLearningController:
            self.right_ctrl = QLearningController(
                side="right", agent=_load_agent("right", agent_path)
            )
        else:
            self.right_ctrl = right_cls(side="right")

        # Create renderer
        if renderer == "TUI":
            self.renderer = TUIRenderer()
        elif renderer == "SDL2":
            self.renderer = SDL2Renderer()
        elif renderer == "log":
            self.renderer = LoggingRenderer()
        else:
            self.renderer = None

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'renderer') and isinstance(self.renderer, SDL2Renderer):
            self.renderer.cleanup()

        print("\nPress Ctrl+C to exit")

    def run(self, fast: bool = False, frame_count: int = None):
        """Run the game loop.
        
        Args:
            fast: If True, skip frame pacing for faster execution (for testing)
            frame_count: If provided, run exactly this many frames when fast mode is enabled
        """
        left_name = self.left_ctrl.__class__.__name__
        right_name = self.right_ctrl.__class__.__name__
        print(f"\nPONG - {left_name} vs {right_name} | Press Ctrl+C to exit")

        try:
            if fast and frame_count is not None:
                for _ in range(frame_count):
                    l_move = self.left_ctrl.get_move(self.physics)
                    r_move = self.right_ctrl.get_move(self.physics)

                    self.physics.update(l_move, r_move)

                    if self.renderer:
                        self.renderer.render(
                            self.physics,
                            self.left_ctrl,
                            self.right_ctrl,
                            l_move,
                            r_move
                        )
            else:
                while getattr(self.left_ctrl, "_running", True):
                    l_move = self.left_ctrl.get_move(self.physics)
                    r_move = self.right_ctrl.get_move(self.physics)

                    self.physics.update(l_move, r_move)

                    if self.renderer:
                        self.renderer.render(
                            self.physics,
                            self.left_ctrl,
                            self.right_ctrl,
                            l_move,
                            r_move
                        )

                    if not fast:
                        time.sleep(0.016)  # ~60 FPS
        except KeyboardInterrupt:
            print("\nGame exited")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pong Game - Terminal Pong with AI")
    parser.add_argument(
        "-l", "--left",
        choices=["human", "algorithmic", "perfect", "nn", "qlearning"],
        default="perfect",
        help="Left paddle controller (default: perfect)",
    )
    parser.add_argument(
        "-r", "--right",
        choices=["human", "algorithmic", "perfect", "nn", "qlearning"],
        default="algorithmic",
        help="Right paddle controller (default: algorithmic)",
    )
    parser.add_argument(
        "--agent",
        default="trained_agent.pkl",
        help="Path to a saved Q-learning agent used by qlearning controllers "
             "(default: trained_agent.pkl)",
    )
    parser.add_argument(
        "--renderer",
        choices=["TUI", "SDL2", "log"],
        default="TUI",
        help="Renderer type (default: TUI)",
    )
    parser.add_argument(
        "--fast",
        type=int,
        default=0,
        help="Run for N frames instead of continuously (for testing)",
    )

    args = parser.parse_args()

    try:
        game = PongGame(
            left_controller=args.left,
            right_controller=args.right,
            renderer=args.renderer,
            agent_path=args.agent,
        )
        game.run(fast=args.fast > 0, frame_count=args.fast if args.fast > 0 else None)
    except Exception as e:
        print(f"Failed to start game: {e}")


if __name__ == "__main__":
    main()
