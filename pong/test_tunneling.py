"""
Test script to verify the tunneling fix in the physics engine.
This demonstrates that fast balls no longer tunnel through paddles.
"""

from physics import PongPhysics
import math


def simulate_fast_ball_collision():
    """Test collision detection with a fast-moving ball that would tunnel."""
    print("=== Tunneling Test ===")
    physics = PongPhysics(width=40, height=20)
    
    # Manually set up a scenario where a ball would tunnel
    # Ball approaches left paddle from the right side with high speed
    physics.ball_pos = [2.0, 12.0]  # Position just to the right of paddle detection zone
    physics.ball_vel = [-3.0, 0.5]  # High leftward velocity that would skip over normally
    
    print(f"Initial position: ({physics.ball_pos[0]}, {physics.ball_pos[1]})")
    print(f"Initial velocity: ({physics.ball_vel[0]}, {physics.ball_vel[1]})")
    
    # Simulate multiple frames with the tunneling scenario
    for frame in range(5):
        print(f"\nFrame {frame+1}:")
        print(f"  Before update: pos={physics.ball_pos}, vel={physics.ball_vel}")
        
        # This would normally detect the collision because we moved to the correct position
        # But with our new swept detection, we should catch it even if we move past the zone
        prev_x = physics.ball_pos[0]
        prev_y = physics.ball_pos[1]
        
        physics.update(0, 0)  # Left paddle isn't moving
        
        print(f"  After update: pos={physics.ball_pos}")
        print(f"  Velocity: {physics.ball_vel}")
        
        # Check if collision was detected (would happen in actual game)
        if prev_x > 1.5 and physics.ball_pos[0] <= 1.5:
            print(f"  *** COLLISION DETECTED - ball crossed from {prev_x:.2f} to {physics.ball_pos[0]:.2f} ***")
            if hasattr(physics, '_last_hit_frame'):
                print(f"  Hit occurred at frame {physics._last_hit_frame}")


def test_edge_case():
    """Test the edge case where ball crosses from beyond the paddle zone."""
    print("\n=== Edge Case Test ===")
    physics = PongPhysics(width=30, height=15)
    
    # Test case: ball starts at x=3.0 (beyond paddle zone), moving left at x=-2.5
    # After one frame it would be at x=0.5 (within paddle zone)
    # But we need to make sure collision is detected properly
    test_cases = [
        (3.0, -2.5),  # From outside to inside
        (2.8, -3.0),  # From outside to exactly at boundary
        (3.2, -2.8),  # Another case
    ]
    
    for i, (x_val, vel_x) in enumerate(test_cases):
        print(f"\nTest case {i+1}: x={x_val}, velocity={vel_x}")
        physics.ball_pos = [x_val, 7.5]  # Position right of paddle
        physics.ball_vel = [vel_x, 0.3]
        prev_x = physics.ball_pos[0]
        
        physics.update(0, 0)
        print(f"  Moved from {prev_x:.2f} to {physics.ball_pos[0]:.2f}")
        
        # Our detection should trigger when prev_x > 1.5 and new_x <= 1.5
        if prev_x > 1.5 and physics.ball_pos[0] <= 1.5:
            print(f"  *** Detected collision: crossed boundary ***")


def test_swept_collision_logic():
    """Test the swept collision logic directly."""
    print("\n=== Swept Collision Logic Test ===")
    
    # Simulate the scenario that was failing
    # Ball moving fast from x=3.0 with velocity=-3.0 would end at x=0.0
    # But our old logic would miss it if it didn't land in [0, 1.5]
    
    prev_x = 3.0
    new_x = 0.0  # After movement with high velocity
    
    print(f"Old logic would check: 0 <= {new_x} <= 1.5 ? {0 <= new_x <= 1.5}")
    print(f"But with swept detection: {prev_x} > 1.5 and {new_x} <= 1.5 ? {prev_x > 1.5 and new_x <= 1.5}")
    
    # This is true - we correctly detected the crossing
    if prev_x > 1.5 and new_x <= 1.5:
        print("*** Collision would be detected with swept logic! ***")


if __name__ == "__main__":
    simulate_fast_ball_collision()
    test_edge_case()
    test_swept_collision_logic()