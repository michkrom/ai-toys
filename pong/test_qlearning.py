"""
Tests for the Q-learning controller: pure state featurization, the replay
pipeline, bounce/score semantics, and end-to-end learning smoke tests.

Fast by design: every test uses small buffers/batches and a fixed random
seed so results are deterministic and reproducible.

Run with:
    pytest test_qlearning.py
    invoke test.qlearning
"""

import copy
import math
import random

import numpy as np
import pytest

from physics import PongPhysics
from qlearning import QLearningAgent
from qlearning.network import QNetwork
from qlearning.replay_buffer import ReplayBuffer
from qlearning.agent import REWARD_BOUNCE, REWARD_WIN, REWARD_LOSS
from qlearning.trainer import (
    SelfPlayTrainer,
    _snapshot,
    bounced_off_my_paddle,
    compute_step_reward,
    point_reward,
)

PADDLE_HALF = PongPhysics().paddle_height / 2  # 2.5
WIDTH, HEIGHT = 80, 25


@pytest.fixture(autouse=True)
def seeded_random():
    """Same seed for every test: deterministic serves, network init and
    numpy samplers."""
    random.seed(0)
    np.random.seed(0)
    yield


def make_agent(**kwargs):
    """Small, fast agent config for tests."""
    kwargs.setdefault("epsilon", 1.0)
    kwargs.setdefault("batch_size", 8)
    kwargs.setdefault("buffer_capacity", 128)
    kwargs.setdefault("target_update_freq", 16)
    return QLearningAgent(side="right", **kwargs)


# ----------------------------------------------------------------------
# State featurization
# ----------------------------------------------------------------------

def test_get_state_is_pure_and_stable():
    """Same physics object -> same vector, no hidden history."""
    p = PongPhysics()
    a = make_agent()
    v1 = a._get_state(p)
    v2 = a._get_state(p)
    assert v1 == v2
    assert len(v1) == 7


def test_get_state_features_bounded():
    """All features stay in [-1, 1] across varied ball states."""
    a = make_agent()
    for _ in range(20):
        p = PongPhysics()
        p.ball_pos = [random.uniform(0.5, WIDTH - 0.5),
                      random.uniform(0.5, HEIGHT - 0.5)]
        p.ball_vel = [random.uniform(-1.5, 1.5),
                      random.uniform(-0.7, 0.7)]
        s = a._get_state(p)
        assert all(-1.0 <= x <= 1.0 for x in s), s


def test_state_mirrored_pov():
    """Both sides see the same world from their own point of view."""
    p = PongPhysics()
    p.ball_pos = [60.0, 10.0]
    r = QLearningAgent(side="right", epsilon=1.0)
    l = QLearningAgent(side="left", epsilon=1.0)
    s_r = r._get_state(p)
    s_l = l._get_state(p)

    # Mirrored x coordinates
    assert s_r[0] == pytest.approx(1.0 - 60.0 / WIDTH)
    assert s_l[0] == pytest.approx(60.0 / WIDTH)

    # Own / opponent paddle centers
    assert s_r[2] == pytest.approx((p.paddle_right + PADDLE_HALF) / HEIGHT)
    assert s_r[3] == pytest.approx((p.paddle_left + PADDLE_HALF) / HEIGHT)
    assert s_l[2] == pytest.approx((p.paddle_left + PADDLE_HALF) / HEIGHT)
    assert s_l[3] == pytest.approx((p.paddle_right + PADDLE_HALF) / HEIGHT)


def test_state_velocity_features_come_from_physics():
    """Velocity is read from ball_vel directly, not inferred from history."""
    p = PongPhysics()
    p.ball_vel = [0.5, 0.5]
    r = QLearningAgent(side="right", epsilon=1.0)
    l = QLearningAgent(side="left", epsilon=1.0)
    s_r = r._get_state(p)
    s_l = l._get_state(p)

    assert s_r[4] == pytest.approx(-0.5 / PongPhysics.MAX_SPEED)
    assert s_l[4] == pytest.approx(0.5 / PongPhysics.MAX_SPEED)
    assert s_r[5] == pytest.approx(0.5 / PongPhysics.MAX_VERTICAL_VELOCITY)
    assert s_l[5] == pytest.approx(0.5 / PongPhysics.MAX_VERTICAL_VELOCITY)

    # Approaching ball has the same mirrored-frame sign on both sides
    p.ball_vel = [0.5, 0.0]  # flying toward the right paddle
    assert r._get_state(p)[4] < 0.0
    assert l._get_state(p)[4] > 0.0


def test_actions_move_the_paddles_they_control():
    """update(left_move, right_move): right-side agent action moves right
    paddle, left-side moves left paddle, opposite paddle stays put."""
    p = PongPhysics()
    p.update(0, 1)
    assert p.paddle_right == pytest.approx(12.5 + 1)
    assert p.paddle_left == pytest.approx(12.5)

    p = PongPhysics()
    p.update(-1, 0)
    assert p.paddle_left == pytest.approx(12.5 - 1)
    assert p.paddle_right == pytest.approx(12.5)


def test_stored_state_matches_action_time_and_differs_from_next():
    """The transition stored in the buffer agrees with the vector the action
    was chosen from, and carries real transition info (s != s')."""
    agent = make_agent(epsilon=0.0)
    p = PongPhysics()

    action = agent.get_action(p)           # network sees s_t
    at_action_time = agent._get_state(p)
    prev = _snapshot(p)
    p.update(0, action)

    agent.store_experience(prev, action, 0.0, p, False)
    state_v, _, _, next_v, _, _ = agent.replay_buffer.buffer[-1]

    assert state_v == at_action_time
    assert state_v != next_v


# ----------------------------------------------------------------------
# Bounce / score detection
# ----------------------------------------------------------------------

def test_bounce_detection_right_paddle():
    p = PongPhysics()
    p.ball_pos = [79.0, 12.5]
    p.ball_vel = [0.5, 0.0]
    p.paddle_right = 10.0                # covers y 10..15; ball at 12.5
    prev = _snapshot(p)
    p.update(0, 0)
    assert p.ball_vel[0] < 0.0, "ball should now fly left"
    assert bounced_off_my_paddle("right", prev, p)
    assert not bounced_off_my_paddle("left", prev, p)


def test_bounce_detection_left_paddle():
    p = PongPhysics()
    p.ball_pos = [0.75, 12.5]
    p.ball_vel = [-0.5, 0.0]
    p.paddle_left = 10.0
    prev = _snapshot(p)
    p.update(0, 0)
    assert p.ball_vel[0] > 0.0
    assert bounced_off_my_paddle("left", prev, p)
    assert not bounced_off_my_paddle("right", prev, p)


def test_wall_bounce_is_not_a_paddle_bounce():
    p = PongPhysics()
    p.ball_pos = [40.0, 24.1]
    p.ball_vel = [-0.5, 0.5]
    prev = _snapshot(p)
    p.update(0, 0)
    assert p.ball_vel[1] < 0.0            # vertical velocity flipped
    assert p.ball_vel[0] == pytest.approx(-0.5)
    assert not bounced_off_my_paddle("right", prev, p)
    assert not bounced_off_my_paddle("left", prev, p)


def test_score_reset_is_not_a_paddle_bounce():
    """A missed ball (score) also flips vx via the serve - the score-change
    guard must prevent it from being misread as a bounce."""
    p = PongPhysics()
    p.ball_pos = [79.5, 5.0]
    p.ball_vel = [0.5, 0.0]
    p.paddle_right = 10.0                 # misses ball at y=5
    prev = _snapshot(p)
    p.update(0, 0)
    assert p.left_score == 1
    assert not bounced_off_my_paddle("right", prev, p)


# ----------------------------------------------------------------------
# Rewards
# ----------------------------------------------------------------------

def test_reward_is_bounce_bonus():
    p = PongPhysics()
    p.ball_pos = [79.0, 12.5]
    p.ball_vel = [0.5, 0.0]
    p.paddle_right = 10.0
    prev = _snapshot(p)
    p.update(0, 0)
    assert compute_step_reward("right", prev, p, 0) == pytest.approx(
        REWARD_BOUNCE
    )


def test_reward_zero_while_ball_moving_away():
    prev = PongPhysics()
    prev.ball_vel = [-0.5, 0.0]           # already flying left, away from the
    curr = _snapshot(prev)                # right-side player -> nothing to do
    curr.ball_pos = [40.0, 12.5]
    assert compute_step_reward("right", prev, curr, 0) == 0.0

    # Same on the other side
    prev.ball_vel = [0.5, 0.0]
    curr = _snapshot(prev)
    curr.ball_pos = [40.0, 12.5]
    assert compute_step_reward("left", prev, curr, 0) == 0.0


def test_reward_tracks_ball_when_inbound():
    prev = PongPhysics()
    curr = _snapshot(prev)
    curr.ball_pos = [42.0, 12.5]
    curr.ball_vel = [0.5, 0.0]            # moving toward right paddle
    curr.paddle_right = 10.0              # paddle center y=12.5: dead under ball
    reward = compute_step_reward("right", prev, curr, 0)
    assert reward == pytest.approx(1.0)   # position reward only, no movement

    # Motion toward the ball adds the movement bonus
    curr.ball_pos = [42.0, 15.0]
    reward = compute_step_reward("right", prev, curr, 1)  # move down toward ball
    assert reward == pytest.approx(1.0 - 2.5 / HEIGHT + 0.5)


# ----------------------------------------------------------------------
# Replay / training pipeline
# ----------------------------------------------------------------------

def test_sample_returns_batch_indices_weights():
    buf = ReplayBuffer(capacity=64, prioritized=True)
    for i in range(24):
        buf.push([i / 24.0] * 7, 1, float(i), [(i + 1) / 24.0] * 7, False,
                 priority=float(i + 1))
    batch, indices, weights = buf.sample(8)
    assert len(batch) == 8
    assert len(indices) == 8
    assert len(weights) == 8
    assert all(0.0 < w <= 1.0 + 1e-9 for w in weights)

    buf2 = ReplayBuffer(capacity=64, prioritized=False)
    for i in range(24):
        buf2.push([i] * 7, 1, 0.0, [i + 1] * 7, False)
    batch2, idx2, w2 = buf2.sample(8)
    assert len(batch2) == 8
    assert all(w == 1.0 for w in w2)


def test_train_step_losses_finite_and_positive():
    agent = make_agent()
    p = PongPhysics()
    losses = []
    for _ in range(40):
        act = agent.get_action(p)
        prev = _snapshot(p)
        p.update(0, act)
        rs = p.right_score > prev.right_score
        ls = p.left_score > prev.left_score
        reward = compute_step_reward("right", prev, p, act)
        if rs:
            reward = REWARD_WIN
        elif ls:
            reward = REWARD_LOSS
        agent.store_experience(prev, act, reward, p, rs or ls)
        loss = agent.train_step()
        if loss:
            losses.append(loss)
    assert len(losses) >= 10
    assert all(math.isfinite(x) and x > 0.0 for x in losses)


def test_store_experience_prioritizes_by_td_error():
    agent = make_agent()
    p = PongPhysics()
    for _ in range(16):
        act = agent.get_action(p)
        prev = _snapshot(p)
        p.update(0, act)
        agent.store_experience(prev, act, 1.0, p, False)
    priorities = list(agent.replay_buffer.priorities)
    assert len(set(round(x, 12) for x in priorities)) > 1


def test_backward_is_weight_scales_gradient():
    """Importance-sampling weight must scale the update linearly (small
    weights so the ±0.5 gradient clip never engages)."""
    net1 = QNetwork(input_size=3, hidden_size=8, output_size=2)
    net1.W1 = np.full_like(net1.W1, 0.01)
    net1.W2 = np.full_like(net1.W2, 0.01)
    net1.b1 = np.zeros(8)
    net1.b2 = np.zeros(2)
    net2 = copy.deepcopy(net1)
    state = [0.5, -0.5, 1.0]
    target = [1.0, 0.0]

    def deltas(net, weight):
        before_w1 = np.copy(net.W1)
        before_w2 = np.copy(net.W2)
        net.backward(state, target, 0.001, weight=weight)
        return (net.W1 - before_w1).ravel(), (net.W2 - before_w2).ravel()

    d1_w1, d1_w2 = deltas(net1, 1.0)
    d2_w1, d2_w2 = deltas(net2, 2.0)
    assert d2_w1 == pytest.approx(2.0 * d1_w1, rel=1e-9)
    assert d2_w2 == pytest.approx(2.0 * d1_w2, rel=1e-9)


def test_selftrain_done_only_on_score():
    """Terminal transitions carry shaped point rewards only - a bounce or a
    dense per-step reward must never be marked as terminal."""
    trainer = SelfPlayTrainer(QLearningAgent, episodes=2, opponent_epsilon=0.05)
    trainer.train(PongPhysics)

    for agent in (trainer.agent, trainer.opponent):
        terminal = [exp for exp in agent.replay_buffer.buffer if exp[4]]
        # each of the 2 episodes ends in exactly one point
        assert len(terminal) == 2
        for exp in terminal:
            assert REWARD_LOSS <= exp[2] <= REWARD_WIN
            assert exp[2] > 0.0 or exp[2] < 0.0   # strictly signed



def test_physics_records_miss_point():
    """Physics records where the ball got past the loser's paddle."""
    p = PongPhysics()
    p.ball_pos = [79.6, 20.0]
    p.ball_vel = [1.4, 0.0]
    p.paddle_right = 0.0                # covers y 0..5, ball passes at y=20
    p.update(0, 0)
    assert p.left_score == 1            # right paddle missed -> left scores
    assert p.last_miss_side == "right"
    assert p.last_miss_y == pytest.approx(20.0)


def test_point_reward_shaped_by_miss_distance():
    """Terminal penalty is proportional to miss distance: near miss is a
    milder punishment than a total whiff."""
    # Agent (right) loses, paddle close to the miss at y=21 (paddle 15..20)
    p = PongPhysics()
    p.ball_pos = [79.6, 21.0]
    p.ball_vel = [1.4, 0.0]
    p.paddle_right = 15.0
    prev = _snapshot(p)
    p.update(0, 0)
    reward, done = point_reward("right", prev, p)
    assert done
    closeness = math.exp(-((21.0 - 17.5) / 2.5) ** 2)
    assert reward == pytest.approx(REWARD_LOSS * (1.0 - closeness))
    assert reward > REWARD_LOSS          # close miss -> milder than -5

    # Paddle far away -> full -5 penalty
    p = PongPhysics()
    p.ball_pos = [79.6, 20.0]
    p.ball_vel = [1.4, 0.0]
    p.paddle_right = 0.0
    prev = _snapshot(p)
    p.update(0, 0)
    reward, done = point_reward("right", prev, p)
    assert done
    assert reward == pytest.approx(REWARD_LOSS)

    # Agent wins when the ball gets past the left paddle:
    # opponent misses far away -> full win bonus
    p = PongPhysics()
    p.ball_pos = [0.4, 20.0]
    p.ball_vel = [-1.4, 0.0]
    p.paddle_left = 0.0
    prev = _snapshot(p)
    p.update(0, 0)
    assert p.right_score == 1
    reward, done = point_reward("right", prev, p)
    assert done
    assert reward == pytest.approx(REWARD_WIN)

    # ...but a near miss by the opponent means a smaller win reward,
    # encouraging angled shots that make the opponent miss wide.
    p = PongPhysics()
    p.ball_pos = [0.4, 10.4]
    p.ball_vel = [-1.4, 0.0]
    p.paddle_left = 5.0                 # covers y 5..10, ball passes at 10.4
    prev = _snapshot(p)
    p.update(0, 0)
    reward, done = point_reward("right", prev, p)
    assert done
    closeness = math.exp(-((10.4 - 7.5) / 2.5) ** 2)
    assert reward == pytest.approx(REWARD_WIN * (0.1 + 0.9 * (1 - closeness)))
    assert reward < REWARD_WIN


def test_agent_learns_to_bounce():
    """Smoke test: the greedy policy bounces the ball more than the early
    random exploration did (fixed seed -> deterministic)."""
    random.seed(11)
    np.random.seed(11)
    agent = make_agent(epsilon=0.9, batch_size=16, buffer_capacity=512)
    first_half = 0
    last_half = 0

    for ep in range(30):
        p = PongPhysics()
        bounces = 0
        for step in range(150):
            act = agent.get_action(p)
            prev = _snapshot(p)
            p.update(0, act)
            if bounced_off_my_paddle("right", prev, p):
                bounces += 1
            rs = p.right_score > prev.right_score
            ls = p.left_score > prev.left_score
            reward = compute_step_reward("right", prev, p, act)
            if rs:
                reward = REWARD_WIN
            elif ls:
                reward = REWARD_LOSS
            agent.store_experience(prev, act, reward, p, rs or ls)
            agent.train_step()
            if rs or ls:
                break
        if ep < 15:
            first_half += bounces
        else:
            last_half += bounces

    assert last_half > first_half


# ----------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------

def test_qlearning_controller_integration():
    """QLearningController feeds the physics object straight to the agent
    and returns valid moves."""
    from controllers import QLearningController

    agent = make_agent(epsilon=0.0)
    ctrl = QLearningController(side="right", agent=agent)
    p = PongPhysics()
    for _ in range(15):
        move = ctrl.get_move(p)
        assert move in (-1, 0, 1)
        p.update(0, move)