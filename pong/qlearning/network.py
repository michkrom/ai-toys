"""
Neural network for Q-value estimation.

A small feedforward network: one ReLU hidden layer, linear output layer.
Backed by numpy for speed (the pure-Python scalar loops were the training
bottleneck, ~16ms per batch vs <1ms now).

Persistence is still pickle-based and backward-compatible: older files that
stored weights as nested Python lists are converted to arrays on load.
"""

import math
import random
import pickle
import os
from typing import List, Tuple

import numpy as np


class QNetwork:
    """
    Simple feedforward network for Q-value estimation.
    """

    def __init__(
        self, input_size: int = 7, hidden_size: int = 64, output_size: int = 3
    ):
        """Initialize network with small random weights (Xavier)."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        limit1 = math.sqrt(6.0 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (hidden_size, input_size))
        limit2 = math.sqrt(6.0 / (hidden_size + output_size))
        self.W2 = np.random.uniform(
            -limit2, limit2, (output_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)

        # Cache for backpropagation
        self.last_h1 = None
        self.last_input = None

    def forward(
        self, state: List[float]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Forward pass through the network.
        Returns: (Q-values, hidden layer activations, hidden layer activations)
        """
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)  # ReLU
        q = self.W2 @ h1 + self.b2

        # Cache for backprop
        self.last_h1 = h1
        self.last_input = x

        return q.tolist(), h1.tolist(), h1.tolist()

    def backward(
            self,
            state: List[float],
            target_q: List[float],
            learning_rate: float = 0.001,
            weight: float = 1.0):
        """
        Backpropagation update using target Q-values (MSE loss).

        weight: importance-sampling weight for prioritized replay; scales
        the per-sample loss contribution. Defaults to 1.0 (unweighted).
        """
        x = np.asarray(state, dtype=np.float64).reshape(-1)
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)
        q = self.W2 @ h1 + self.b2
        t = np.asarray(target_q, dtype=np.float64).reshape(-1)

        # Output layer gradients (MSE derivative: q - target), IS-weighted
        delta2 = weight * (q - t)

        # Hidden layer gradients (ReLU derivative is 1 where h1 > 0), from
        # the ORIGINAL W2 - the standard backprop formulation.
        delta_h1 = (self.W2.T @ delta2) * (h1 > 0)

        # Gradient clipping on the lr-scaled weight updates (as before)
        self.W2 -= learning_rate * np.clip(np.outer(delta2, h1), -0.5, 0.5)
        self.b2 -= learning_rate * delta2
        self.W1 -= learning_rate * np.clip(np.outer(delta_h1, x), -0.5, 0.5)
        self.b1 -= learning_rate * delta_h1

    def copy_weights(self, other_network):
        """Copy weights from another network (for target network)."""
        self.W1 = np.copy(other_network.W1)
        self.W2 = np.copy(other_network.W2)
        self.b1 = np.copy(other_network.b1)
        self.b2 = np.copy(other_network.b2)

    def save(self, filepath: str):
        """Save network weights to file."""
        data = {
            "W1": self.W1,
            "W2": self.W2,
            "b1": self.b1,
            "b2": self.b2,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Load network weights from file (accepts legacy list-based files)."""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.W1 = np.asarray(data["W1"], dtype=np.float64)
            self.W2 = np.asarray(data["W2"], dtype=np.float64)
            self.b1 = np.asarray(data["b1"], dtype=np.float64)
            self.b2 = np.asarray(data["b2"], dtype=np.float64)
            self.input_size = data["input_size"]
            self.hidden_size = data["hidden_size"]
            self.output_size = data["output_size"]