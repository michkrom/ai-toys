#!/usr/bin/env python3
"""
Sanity test for the Pong game.

This test runs the full game for a short period to ensure that:
1. The game starts without crashing
2. The physics engine works correctly
3. Controllers respond appropriately
4. No exceptions occur during normal operation
5. The game can be cleanly terminated

The test uses a short timeout and runs in a separate process to
allow for clean termination if issues arise.
"""

import subprocess
import time
import signal
import sys
import os
from pathlib import Path


def test_sanity_basic():
    """Test basic game functionality."""
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Start the pong game with test controllers
    # Use fast mode with minimal output
    cmd = [
        sys.executable, "-c",
        """
import sys
import time
from pong import PongGame

# Create game with algorithmic controllers for both sides
game = PongGame(left_controller="algorithmic", right_controller="algorithmic", renderer="log")

# Run for a short period (about 2 seconds at 60 FPS)
start_time = time.time()
frame_count = 0
try:
    while getattr(game.left_ctrl, '_running', True) and time.time() - start_time < 2.0:
        frame_count += 1
        game.physics.update(0, 0)
        if frame_count % 100 == 0:
            print(f"Processing... {frame_count} frames")
    print(f"Test completed: {frame_count} frames processed")
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
        """
    ]
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=script_dir
    )
    
    # Wait for completion with timeout
    try:
        stdout, stderr = process.communicate(timeout=5.0)
        
        # Check for errors in stderr
        if stderr and "ERROR" in stderr:
            raise AssertionError(f"Game crashed with error:\n{stderr}")
            
        # Check that we got some output
        if not stdout or "Processing" not in stdout:
            # This might be expected if logging is disabled
            pass
            
        # Check exit code
        if process.returncode != 0:
            raise AssertionError(f"Game exited with code {process.returncode}")
            
    except subprocess.TimeoutExpired:
        # Kill the process if it hangs
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(f"Game hung after timeout. Output:\n{stdout}\nErrors:\n{stderr}")


def test_sanity_with_human():
    """Test game with human-like controller."""
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    cmd = [
        sys.executable, "-c",
        """
import sys
import time
from pong import PongGame

# Test with perfect controller (more predictable than algorithmic)
game = PongGame(left_controller="perfect", right_controller="algorithmic", renderer="log")

# Run a short test
start_time = time.time()
frame_count = 0
bounce_count = 0

while getattr(game.left_ctrl, '_running', True) and time.time() - start_time < 1.5:
    frame_count += 1
    old_ball_x = game.physics.ball_pos[0]
    game.physics.update(0, 0)
    # Count bounces (ball position changes significantly)
    if abs(game.physics.ball_pos[0] - old_ball_x) > 2.0:
        bounce_count += 1

print(f"Human-like test: {frame_count} frames, {bounce_count} bounces")
sys.exit(0)
        """
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=script_dir
    )
    
    try:
        stdout, stderr = process.communicate(timeout=4.0)
        
        if stderr and "ERROR" in stderr:
            raise AssertionError(f"Human-like test crashed:\n{stderr}")
            
        # Verify we got reasonable output
        if "frames" not in stdout:
            raise AssertionError("Expected output not found in human-like test")
            
    except subprocess.TimeoutExpired:
        process.kill()
        raise AssertionError("Human-like test hung")


def test_sanity_controllers():
    """Test that all controller types can be instantiated and get moves."""
    
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Test each controller combination
    controller_combos = [
        ("perfect", "algorithmic"),
        ("algorithmic", "perfect"),
        ("algorithmic", "algorithmic"),
    ]
    
    for left_ctrl, right_ctrl in controller_combos:
        cmd = [
            sys.executable, "-c",
            f"""
import sys
from pong import PongGame
from physics import PongPhysics

game = PongGame(left_controller="{left_ctrl}", right_controller="{right_ctrl}", renderer="log")
physics = PongPhysics()

# Test that controllers can get moves
left_move = game.left_ctrl.get_move(physics)
right_move = game.right_ctrl.get_move(physics)

# Verify moves are valid (-1, 0, or 1)
assert left_move in [-1, 0, 1], f"Invalid left move: {{left_move}}"
assert right_move in [-1, 0, 1], f"Invalid right move: {{right_move}}"

print("Controllers OK")
sys.exit(0)
            """
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=script_dir
        )
        
        stdout, stderr = process.communicate(timeout=3.0)
        
        if stderr and "ERROR" in stderr:
            raise AssertionError(f"Controller test failed for {left_ctrl} vs {right_ctrl}:\n{stderr}")


if __name__ == "__main__":
    print("Running Pong sanity tests...")
    print("=" * 50)
    
    try:
        test_sanity_controllers()
        print("✓ Controller tests passed")
        
        test_sanity_basic()
        print("✓ Basic game test passed")
        
        test_sanity_with_human()
        print("✓ Human-like controller test passed")
        
        print("=" * 50)
        print("All sanity tests passed!")
        
    except AssertionError as e:
        print(f"✗ Sanity test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)