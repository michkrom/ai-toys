#!/usr/bin/env python3
"""
Unit tests for wall-bouncing edge cases.

Verifies that the ball always bounces off the top and bottom walls, even when
its vertical velocity is very small (near-zero). A regression earlier caused
the ball to get 'stuck' hovering at the clamp boundary (y=0.5 or y=height-0.5)
because the bounce only fired when the next step fully crossed y=0 / y=height,
which a tiny velocity could never do.
"""

import unittest

from physics import PongPhysics

# The position clamp used by the physics engine.
MIN_Y = 0.5
MAX_Y = PongPhysics().height - 0.5


class TestWallBounce(unittest.TestCase):
    def setUp(self):
        self.width = 40
        self.height = 20
        self.p = PongPhysics(width=self.width, height=self.height)

    def test_small_velocity_bounces_off_top(self):
        """A ball at the top with tiny upward velocity must bounce down, not hover."""
        p = self.p
        p.ball_pos = [20.0, MIN_Y]
        p.ball_vel = [0.4, -0.075]  # Small upward velocity at the top edge

        p.update(0, 0)

        # Position must escape the wall and velocity must now point downward.
        self.assertGreater(p.ball_pos[1], MIN_Y)
        self.assertGreater(p.ball_vel[1], 0.0)

    def test_small_velocity_bounces_off_bottom(self):
        """A ball at the bottom with tiny downward velocity must bounce up, not hover."""
        p = self.p
        p.ball_pos = [20.0, MAX_Y]
        p.ball_vel = [0.4, 0.075]  # Small downward velocity at the bottom edge

        p.update(0, 0)

        self.assertLess(p.ball_pos[1], MAX_Y)
        self.assertLess(p.ball_vel[1], 0.0)

    def test_ball_escapes_top_each_frame_after_bounce(self):
        """Once near the top, the ball must keep moving away over several frames."""
        p = self.p
        p.ball_pos = [20.0, MIN_Y]
        p.ball_vel = [0.4, -0.075]

        ys = []
        for _ in range(10):
            p.update(0, 0)
            ys.append(p.ball_pos[1])
            # Must never be stuck on the clamp boundary.
            self.assertNotAlmostEqual(p.ball_pos[1], MIN_Y, places=6)

        # y should monotonically increase while moving away from the top wall.
        self.assertTrue(all(b - a > 0 for a, b in zip(ys, ys[1:])),
                        f"ball should move away from top, got y={ys}")

    def test_ball_escapes_bottom_each_frame_after_bounce(self):
        """Once near the bottom, the ball must keep moving away over several frames."""
        p = self.p
        p.ball_pos = [20.0, MAX_Y]
        p.ball_vel = [0.4, 0.075]

        ys = []
        for _ in range(10):
            p.update(0, 0)
            ys.append(p.ball_pos[1])
            self.assertNotAlmostEqual(p.ball_pos[1], MAX_Y, places=6)

        self.assertTrue(all(b - a < 0 for a, b in zip(ys, ys[1:])),
                        f"ball should move away from bottom, got y={ys}")
        # Leaving the bottom with an upward velocity (y decreases = negative vy).
        self.assertLessEqual(p.ball_vel[1], 0.0)

    def test_no_postion_never_goes_out_of_bounds(self):
        """The ball position must always stay within the playfield height."""
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for sx, sy in directions:
            p = PongPhysics(width=self.width, height=self.height)
            p.ball_pos = [20.0, 10.0]
            p.ball_vel = [0.5 * sx, 0.5 * sy]
            for _ in range(500):
                p.update(0, 0)
                self.assertGreaterEqual(p.ball_pos[1], 0.0,
                                        "y below 0 (top out of bounds)")
                self.assertLessEqual(p.ball_pos[1], self.height,
                                     "y above height (bottom out of bounds)")

    def test_vertical_velocity_never_zero_after_bounce_or_reset(self):
        """Vertical velocity must never be exactly zero after serving or wall bounce."""
        p = self.p
        # Exercise the server reset path.
        p._reset_ball(1)
        self.assertNotEqual(p.ball_vel[1], 0.0)
        p._reset_ball(-1)
        self.assertNotEqual(p.ball_vel[1], 0.0)

        # Drive the ball repeatedly against both walls with small velocity.
        p.ball_pos = [20.0, MIN_Y]
        p.ball_vel = [0.4, -0.075]
        for _ in range(200):
            p.update(0, 0)
            self.assertNotEqual(p.ball_vel[1], 0.0,
                                "vertical velocity became zero")

    def test_top_bounce_inverts_vertical_keeps_horizontal(self):
        """A top bounce must flip vertical velocity sign but leave horizontal unchanged."""
        p = self.p
        p.ball_pos = [20.0, MIN_Y]
        p.ball_vel = [0.4, -0.5]  # moving right and up toward the top

        p.update(0, 0)

        # Horizontal is untouched.
        self.assertEqual(p.ball_vel[0], 0.4)
        # Vertical is inverted: was -0.5 (up), now +0.5 (down), same magnitude.
        self.assertEqual(p.ball_vel[1], 0.5)

    def test_bottom_bounce_inverts_vertical_keeps_horizontal(self):
        """A bottom bounce must flip vertical velocity sign but leave horizontal unchanged."""
        p = self.p
        p.ball_pos = [20.0, MAX_Y]
        p.ball_vel = [0.4, 0.5]  # moving right and down toward the bottom

        p.update(0, 0)

        self.assertEqual(p.ball_vel[0], 0.4)
        # Vertical is inverted: was +0.5 (down), now -0.5 (up), same magnitude.
        self.assertEqual(p.ball_vel[1], -0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
