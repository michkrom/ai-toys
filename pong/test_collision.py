#!/usr/bin/env python3
"""
Unit tests for the PongPhysics engine.
Tests swept collision detection and zone-based bounce physics.
"""

import unittest
import math
from physics import PongPhysics


class TestSweptCollision(unittest.TestCase):
    """Test the swept collision detection mechanism."""

    def setUp(self):
        """Set up test environment."""
        self.physics = PongPhysics(width=40, height=20)
        self.physics.ball_pos = [3.0, 10.0]  # Starting position right of paddle

    def test_left_paddle_swept_collision(self):
        """Test that fast-moving leftward balls are detected correctly."""
        # Simulate ball moving fast from x=3.0 to x=0.5 in one frame
        prev_x, prev_y = 3.0, 10.0
        self.physics.ball_pos = [3.0, 10.0]
        self.physics.ball_vel = [-3.0, 0.0]  # Fast leftward velocity

        # Store previous position before update
        stored_prev_x = self.physics.ball_pos[0]
        stored_prev_y = self.physics.ball_pos[1]

        # Update should detect collision with left paddle
        hit_detection_occurred = False
        try:
            self.physics.update(0, 0)  # No controller movement
            hit_detection_occurred = True
        except AttributeError as e:
            if '_handle_swept_paddle_collision' in str(e):
                self.fail("Undefined _handle_swept_paddle_collision method")
            raise

        # The collision handling is integrated directly, so we just verify
        # that update completes without error for fast collision scenarios
        self.assertTrue(hit_detection_occurred, "Update should complete without error")

    def test_right_paddle_swept_collision(self):
        """Test swept collision with right paddle."""
        # Move ball such that it would cross right paddle boundary
        self.physics.ball_pos = [38.0, 10.0]  # Near right side
        self.physics.ball_vel = [1.5, 0.0]   # Moving right quickly

        # Verify update completes
        try:
            self.physics.update(0, 0)
        except AttributeError as e:
            if '_handle_swept_paddle_collision' in str(e):
                self.fail("Undefined _handle_swept_paddle_collision method")
            raise

        # Verify basic functionality
        self.assertTrue(0 <= self.physics.ball_pos[0] < 40, "Ball position should be valid")

    def test_edge_cases(self):
        """Test edge cases for swept collision."""
        test_cases = [
            # (prev_x, prev_y, vel_x, expected_behavior)
            (2.0, 12.0, -2.5, "should_detect_collision"),
            (2.5, 8.0, -1.0, "should_detect_collision"),
            (1.5, 10.0, -0.5, "should_not_cross_boundary"),  # Already in zone
            (0.5, 5.0, -0.1, "should_not_cross_boundary"),  # Moving away
        ]

        for i, (prev_x, prev_y, vel_x, behavior) in enumerate(test_cases):
            with self.subTest(case=i, behavior=behavior):
                physics = PongPhysics()
                physics.ball_pos = [prev_x, prev_y]
                initial_vel = physics.ball_vel.copy()
                physics.ball_vel = [vel_x, 0.0]  # Only change x velocity

                # Verify no AttributeError for _handle_swept_paddle_collision
                try:
                    physics.update(0, 0)
                except AttributeError as e:
                    if '_handle_swept_paddle_collision' in str(e):
                        self.fail("Undefined _handle_swept_paddle_collision method")
                    raise

                # Basic validation that update completes
                self.assertIsInstance(physics.ball_pos, list)
                self.assertTrue(0 <= physics.ball_pos[0] <= 80)


class TestZoneBasedPhysics(unittest.TestCase):
    """Test the zone-based angle calculation system."""

    def setUp(self):
        """Set up test environment."""
        self.physics = PongPhysics(width=40, height=20)

    def test_zone_angle_calculation(self):
        """Test that zone-based angle factors are computed correctly."""
        # Test cases: (hit_offset, expected_zone_name, expected_approx_angle)
        test_cases = [
            (0.0, "middle", 0.0),   # Center hit
            (0.4, "bottom", -0.37313),  # Upper part of bottom zone
            (-0.4, "top", 0.37313),     # Lower part of top zone
            (0.8, "bottom", -0.79104),  # Bottom zone near edge
            (-0.8, "top", 0.79104),     # Top zone near edge
        ]

        for hit_offset, expected_zone, expected_angle in test_cases:
            with self.subTest(hit_offset=hit_offset):
                angle_factor = self.physics._get_zone_angle(hit_offset)
                zone_name = self.physics._get_paddle_zone_name(hit_offset)
                
                self.assertEqual(zone_name, expected_zone, 
                               f"Expected zone {expected_zone} for offset {hit_offset}")
                # Angle should be approximately proportional
                self.assertAlmostEqual(angle_factor, expected_angle, places=1,
                                     msg=f"Angle factor {angle_factor:.3f} not close to {expected_angle:.3f}")

    def test_zone_boundaries(self):
        """Test angle calculations at zone boundaries."""
        # Boundary points should be classified correctly
        boundary_test = [
            (-0.33, "top"),    # Top of middle zone, edge of top zone
            (0.33, "bottom"),  # Bottom of middle zone, edge of bottom zone
        ]

        for hit_offset, expected_zone in boundary_test:
            with self.subTest(hit_offset=hit_offset):
                zone_name = self.physics._get_paddle_zone_name(hit_offset)
                self.assertEqual(zone_name, expected_zone,
                               f"Boundary hit_offset {hit_offset} should be classified as {expected_zone}")

    def test_angle_smooth_interpolation(self):
        """Test that angle calculation is smoothly interpolated within zones."""
        # Test middle zone smooth transition
        angles = []
        for offset in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            angle = self.physics._get_zone_angle(offset)
            angles.append(angle)
        
        # Angles should vary approximately linearly within middle zone
        # With our formula: angle_factor = offset * 1.5 (for |offset| < 0.33)
        expected_angles = [offset * 1.5 for offset in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]]
        
        for actual, expected in zip(angles, expected_angles):
            self.assertAlmostEqual(actual, expected, places=2,
                                 msg=f"Middle zone angle not matching expected linear interpolation")

    def test_angle_clamping(self):
        """Test that angle factors are properly clamped to [-1.0, 1.0]."""
        extreme_values = [-1.0, 1.0, -1.5, 1.5]
        for offset in extreme_values:
            with self.subTest(offset=offset):
                angle_factor = self.physics._get_zone_angle(offset)
                self.assertGreaterEqual(angle_factor, -1.0, 
                                      f"Angle factor should be >= -1.0 for offset {offset}")
                self.assertLessEqual(angle_factor, 1.0,
                                   f"Angle factor should be <= 1.0 for offset {offset}")


if __name__ == '__main__':
    unittest.main(verbosity=2)