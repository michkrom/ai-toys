#!/home/m/.venv/bin/python3
"""
Test script for Pong physics engine with zone-based paddle bouncing.
Uses LoggingRenderer to observe ball behavior.
"""

from physics import PongPhysics
from controllers import QLearningController, AlgorithmicController
from renderers import LoggingRenderer
import time


def test_physics_directly():
    """Test physics engine directly without controllers."""
    print("=== Direct Physics Engine Test ===")
    physics = PongPhysics(width=40, height=20)
    
    # Initialize with a diagonal trajectory
    physics.ball_vel = [0.5, 0.5]  # Moving up and right
    physics.ball_pos = [5.0, 10.0]   # Starting position
    
    print(f"Initial state: pos={physics.ball_pos}, vel={physics.ball_vel}")
    
    # Simulate some frames to see collision behavior
    for frame in range(20):
        # Move ball
        physics.update(0, 0)  # No paddle movement
        
        # Check if we hit a paddle
        if 0 <= physics.ball_pos[0] <= 1.5:  # Left paddle zone
            print(f"Frame {frame+1}: Left paddle collision!")
            print(f"  Ball position: ({physics.ball_pos[0]:.1f}, {physics.ball_pos[1]:.1f})")
            print(f"  Ball velocity: ({physics.ball_vel[0]:.2f}, {physics.ball_vel[1]:.2f})")
            
        # Continue simulation
        time.sleep(0.1)
    
    return True


def test_zone_behavior():
    """Test the zone-based angle calculation."""
    print("\n=== Zone Behavior Test ===")
    physics = PongPhysics(width=40, height=20)
    
    # Test different hit positions on the left paddle
    test_cases = [
        (0.0, "center"),
        (1.0, "upper-middle"),
        (2.0, "top"),
        (-1.0, "lower-middle"),
        (-2.0, "bottom")
    ]
    
    paddle_y = physics.paddle_left
    paddle_height = physics.paddle_height
    paddle_center = paddle_y + paddle_height / 2
    
    for offset, description in test_cases:
        # Calculate hit position
        test_y = paddle_center + offset
        hit_offset = (test_y - paddle_center) / (paddle_height / 2)
        hit_offset = max(-1.0, min(1.0, hit_offset))  # Clamp to [-1, 1]
        
        # Get zone and angle factor
        zone_name = physics._get_paddle_zone_name(hit_offset)
        angle_factor = physics._get_zone_angle(hit_offset)
        
        print(f"{description:>12}: offset={offset:>4.1f}, norm={hit_offset:>5.2f}, "
              f"zone={zone_name:>8}, angle_factor={angle_factor:>6.3f}")


if __name__ == "__main__":
    test_physics_directly()
    test_zone_behavior()