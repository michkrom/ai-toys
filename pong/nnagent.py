"""
Neural Network Agent for Pong.

A simple neural network agent that learns through backpropagation.
"""

import random
import math


class NNAgent:
    """Neural network agent for Pong."""

    def __init__(self, input_size: int = 8, h1_size: int = 16, h2_size: int = 16):
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(h1_size)]
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(h1_size)] for _ in range(h2_size)]
        self.W3 = [[random.uniform(-0.1, 0.1) for _ in range(h2_size)] for _ in range(3)]

        self.learning_rate = 0.01
        self.gamma = 0.95
        self.episode_memory = []

    def _softmax(self, x):
        """Softmax activation for output layer."""
        max_x = max(x)
        e_x = [math.exp(val - max_x) for val in x]
        sum_e_x = sum(e_x)
        return [val / sum_e_x for val in e_x]

    def _sigmoid(self, x):
        """Sigmoid activation."""
        x = max(-20, min(20, x))
        return 1 / (1 + math.exp(-x))

    def _relu(self, x):
        """ReLU activation."""
        return max(0, x)

    def _forward(self, state):
        """Forward pass through network."""
        h1 = []
        for i in range(len(self.W1)):
            z = sum(state[j] * self.W1[i][j] for j in range(len(state)))
            h1.append(self._relu(z))
        
        h2 = []
        for i in range(len(self.W2)):
            z = sum(h1[j] * self.W2[i][j] for j in range(len(h1)))
            h2.append(self._relu(z))
        
        logits = [sum(h2[j] * self.W3[i][j] for j in range(len(h2))) for i in range(3)]
        probs = self._softmax(logits)
        return probs, h1, h2

    def _backprop(self, state, h1, h2, action, reward):
        """Backpropagation update."""
        probs, _, _ = self._forward(state)

        # Calculate target
        if action == 1:
            target = 1
        else:
            target = 0

        _, _, logits = self._forward(state)
        error = probs[action] - target

        # Update weights (simplified)
        delta_w3 = [[h2[j] * error * self.learning_rate for j in range(len(h2))] for _ in range(3)]
        delta_h2 = [error * self.W3[i][action] for i in range(len(self.W3))]
        delta_w2 = [[h1[j] * delta_h2[i] * self.learning_rate for j in range(len(h1))] for i in range(len(self.W2))]
        delta_h1 = [sum(delta_h2[i] * self.W2[i][j] for i in range(len(self.W2))) for j in range(len(h1))]
        delta_w1 = [[state[j] * delta_h1[i] * self.learning_rate for j in range(len(state))] for i in range(len(self.W1))]

        # Apply updates
        for i in range(len(self.W1)):
            for j in range(len(self.W1[i])):
                self.W1[i][j] += delta_w1[i][j]
        for i in range(len(self.W2)):
            for j in range(len(self.W2[i])):
                self.W2[i][j] += delta_w2[i][j]
        for i in range(len(self.W3)):
            for j in range(len(self.W3[i])):
                self.W3[i][j] += delta_w3[i][j]

    def _record_step(self, state, action, h1, h2):
        """Record a step for later backpropagation."""
        self.episode_memory.append((state, action, h1, h2))

    def _clear_memory(self):
        """Clear episode memory."""
        self.episode_memory = []

    def get_action(self, state) -> int:
        """Get action based on current state."""
        probs, h1, h2 = self._forward(state)
        action_idx = random.choices([0, 1, 2], weights=probs)[0]
        action = action_idx - 1  # Map to -1, 0, 1
        self._record_step(state, action_idx, h1, h2)
        return action

    def on_event(self, event_type: str, reward: float):
        """Handle game event and train."""
        rewards = {
            'bounce': 0.1,
            'point_win': 1.0,
            'point_loss': -1.0,
        }
        base_reward = rewards.get(event_type, 0)

        # Backpropagate with discounted rewards
        for i in range(len(self.episode_memory) - 1, -1, -1):
            state, action, h1, h2 = self.episode_memory[i]
            steps_back = len(self.episode_memory) - i
            discounted_reward = base_reward * (self.gamma ** steps_back)
            self._backprop(state, h1, h2, action, discounted_reward)

        self._clear_memory()