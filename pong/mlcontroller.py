"""
Machine Learning Controller for Pong.
"""

import random


class MLController:
    """ML-based controller wrapper."""

    def __init__(self, side: str = "left"):
        self.side = side
        self._model = None

    def get_move(self, game) -> int:
        """Get move from ML model."""
        # Simple placeholder - would use actual ML model
        return random.choice([-1, 0, 1])

    def load(self, path: str):
        """Load model from file."""
        pass