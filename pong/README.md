# Classic TUI Pong Game

A simple, cross-platform terminal-based Pong game with multiple AI controllers,
including a Deep Q-Network (DQN) self-learning agent.

## Features
- ✅ Classic Pong gameplay in terminal
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Human vs Computer AI opponent
- ✅ Self-learning AI using Deep Q-Network (Double DQN + prioritized replay)
- ✅ Multiple AI difficulty levels
- ✅ Angle-based ball bouncing physics
- ✅ Clean rendering with center dotted line
- ✅ Fast, reproducible headless training + live bounces/W-L monitoring

## Controls
- **W** or **↑** - Move left paddle up
- **S** or **↓** - Move left paddle down
- **Q** - Quit game

## Quick Start

```bash
# Dependencies
pip install -r requirements.txt          # adds request: invoke for the task runner

# Run the game (perfect vs algorithmic by default)
python3 pong.py

# Train a Q-learning agent (headless, ~a minute), then play against it
python3 train_qlearning.py --episodes 100 --interval 15 --opponent algorithmic \
    --seed 11 --save trained_agent.pkl
python3 pong.py -l algorithmic -r qlearning --agent trained_agent.pkl
```

## Running the game

```bash
# Default: perfect vs algorithmic
python3 pong.py

# Human vs Q-Learning AI (uses a saved agent)
python3 pong.py -l human -r qlearning --agent trained_agent.pkl

# Algorithmic vs Q-Learning AI
python3 pong.py -l algorithmic -r qlearning

# Use SDL2 renderer
python3 pong.py --renderer SDL2

# Run exactly 1000 frames headless (no pacing), useful for testing / log analysis
python3 pong.py --renderer log --fast 1000
```

| Option | Choices | Default | Description |
|--------|---------|---------|-------------|
| `-l`, `--left` | human, algorithmic, perfect, nn, qlearning | perfect | Left paddle controller |
| `-r`, `--right` | human, algorithmic, perfect, nn, qlearning | algorithmic | Right paddle controller |
| `--agent` | path | trained_agent.pkl | Saved Q-learning agent used by `qlearning` controllers |
| `--renderer` | TUI, SDL2, log | TUI | Renderer type (`log` = headless state logging) |
| `--fast` | int | 0 | Run exactly N frames and exit (0 = run continuously) |

> A `qlearning` controller loads the saved agent and plays greedily. If the
> agent file is missing it warns and falls back to random play (so a "qlearning"
> player that never improves usually means no trained agent was loaded).

## Task runner (invoke)

All the common workflows are wrapped in an [invoke](https://www.pyinvoke.org/)
`tasks.py`:

```bash
invoke test                 # full pytest suite
invoke test.qlearning       # Q-learning tests only
invoke test.smoke           # headless game smoke run

invoke train.qlearning --episodes 200 --interval 25 --opponent algorithmic
invoke train.selfplay --episodes 200
invoke train.perfect --episodes 100

invoke play --right qlearning   # TUI game with the saved agent
invoke clean                    # remove caches
```

## Self-Learning AI (Q-learning)

### State representation (7 normalized features)

The network input is a **pure function of the physics state** - no hidden
history, so the state stored in the replay buffer is always identical to the
one the action was chosen from. All features are from the controller's own
point of view (mirrored so the same weights work for either side):

| # | Feature | Notes |
|---|---------|-------|
| 0 | `ball_x` | 0 at my paddle, 1 at the far wall (mirrored) |
| 1 | `ball_y` | |
| 2 | `my_paddle_center` | |
| 3 | `opponent_paddle_center` | |
| 4 | `ball_vx` | signed, in the same mirrored frame |
| 5 | `ball_vy` | signed |
| 6 | `paddle_offset` | `ball_y - my_paddle_center` |

Velocity is read directly from `physics.ball_vel` rather than reconstructed
from a position history - the history proxy is only meaningful around bounces
and is corrupted at score resets, so it's useless as a general feature.

### Actions
- `-1` = Move paddle up
- `0` = Stay in place
- `1` = Move paddle down

### Rewards
- `+5` for bouncing the ball off your own paddle
- `+10 · (0.1 + 0.9·(1 – opponent_closeness))` for winning a point (encourages
  angled shots that make the opponent miss wide)
- `-5 · (1 – my_closeness)` for losing a point, where
  `closeness = exp(-(miss_distance / paddle_half)²)` - the penalty is
  **proportional to how close the paddle was** to the ball when it got past,
  so the Q-function has a gradient right at the failure event
- Dense tracking reward (`0..1` for being under the ball while it's inbound,
  plus `+0.5` for moving toward it)

### Algorithm (`qlearning/`)
- **Double DQN** - the online net selects the next action, the target net
  evaluates it (reduces Q-overestimation)
- **Prioritized experience replay** - stored transitions carry a TD-error
  priority, and the loss is weighted by importance-sampling weights
- **Target network** synced every `target_update_freq` steps
- **Numpy-backed MLP** (7→64→3, ReLU) for fast training

### Correctness fixes baked in
- `physics.update(left_move, right_move)`: action→paddle mapping is respected
  everywhere (the right-side agent's action moves the *right* paddle).
- Physics is **snapshotted before every `update`** - it mutates in place, so
  without a snapshot the stored (state, next_state) pair aliases and carries no
  transition.
- `done` is set **only on a point being scored** - a bounce is never terminal
  (marking it terminal would teach the agent that rallies end at bounces).

## Training & monitoring

### `train_qlearning.py` - train against a fixed opponent

```bash
python3 train_qlearning.py --episodes 150 --interval 25 --epsilon 0.9 \
    --opponent algorithmic --save trained_agent.pkl --seed 11

# Live table: watch bounces/ep climb and W% climb as epsilon falls
```

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 100 | Number of training episodes |
| `--interval` | 25 | Report bounces / W / L / rally every N episodes |
| `--epsilon` | 0.8 | Initial exploration rate |
| `--opponent` | stationary | `stationary`, `algorithmic`, or `perfect` |
| `--save` | (none) | Save the trained agent to this path |
| `--seed` | (random) | Seed for reproducible runs |
| `--test-steps` | 20 | Greedy test roll-out length after training |

Output columns: `bounces/ep` (how often the agent returns the ball), `W`/`L`
(points won/lost per interval), `W%`, `rally` (avg points fought per rally),
`eps`.

### `train_agent.py` - self-play or vs perfect

```bash
python3 train_agent.py --method selfplay --episodes 200   # two agents train
python3 train_agent.py --method perfect --episodes 100    # vs the perfect AI
python3 train_agent.py --load --save-path trained_agent.pkl
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | selfplay | `selfplay` or `perfect` |
| `--episodes` | 500 | Number of training episodes |
| `--opponent-epsilon` | 0.1 | Opponent exploration in self-play |
| `--load` | - | Load and play a saved agent |
| `--evaluate` | 0 | Evaluation episodes after training |
| `--save-path` | trained_agent.pkl | Where to save the agent |

### Reading the learning signal - set your expectations

- **`stationary` / `algorithmic`** are winnable: expect `W%` to climb toward
  ~90-100% over the first few intervals. Untrained play is 0 bounces / 0 wins.
- **`perfect`** is designed to never miss, so **W stays 0 by construction** -
  there the signal is `bounces/ep` and `rally` (a trained agent typically
  returns the ball 4-14× longer than an untrained one).
- Training is **seed-sensitive** (a tiny DQN can occasionally freeze into a
  "never hit the ball" attractor). If bounces stay at 0 / rally at 80 across an
  interval, restart with a different `--seed` (11 and 42 are known-good vs
  algorithmic). Use `--save` with a good seed for `pong.py`.

## Files
- `pong.py` - Main game entry point
- `physics.py` - Game physics engine (also records `last_miss_side/y` for reward shaping)
- `renderers.py` - TUI, SDL2 and Log renderers
- `controllers.py` - Controllers (Human, Algorithmic, Perfect, NN, QLearning)
- `qlearning/` - DQN package: `agent.py`, `network.py`, `replay_buffer.py`, `trainer.py`
- `train_qlearning.py` - Train vs fixed opponents with monitoring
- `train_agent.py` - Self-play / vs-perfect training CLI
- `qlearning_agent.py` - Legacy re-export wrapper (kept for compatibility)
- `tasks.py` - Invoke task runner
- `test_qlearning.py` - Q-learning unit tests (state, replay, rewards, learning)
- `test_physics.py`, `test_collision.py`, `test_tunneling.py`,
  `test_sanity.py`, `test_wall_bounce.py` - Physics / controller regression tests
- `README.md` - This documentation

## Controllers

| Controller | Description |
|------------|-------------|
| `human` | Human player |
| `algorithmic` | Human-like AI with prediction and reaction delay |
| `perfect` | Predicts intercept (incl. wall bounces); never misses its side |
| `nn` | Basic neural-network placeholder controller |
| `qlearning` | Self-learning DQN agent (uses the saved `--agent` model) |

## Testing

```bash
pip install -r requirements.txt
pytest -q          # or: invoke test
```

`test_qlearning.py` covers: pure/stateless state featurization, POV mirroring,
action→paddle causality, real vs. fake bounce detection, miss-distance point
rewards, replay-buffer sampling/priorities, IS-weighted backprop, and a seeded
learning smoke test. The legacy physics tests cover collisions, tunneling and
wall bounces.

## Physics

The physics engine implements classic Pong mechanics with predictive, tunnel-safe
(swept) collision detection:
- Ball bounces off walls and paddles; the bounce angle depends on where on the
  paddle the ball hits (zone-based)
- Wall bounces invert vertical velocity while preserving horizontal velocity
- Ball speed is bounded (min/max) so long rallies stay stable
- After a point, the ball is served toward the opponent

## Future Improvements
- [ ] Keep-best checkpointing during training (robustness vs. seed-variance/freeze)
- [ ] Curriculum learning (start easy, increase difficulty)
- [ ] Noisy-Net / Rainbow-DQN enhancements
- [ ] Transfer learning between agents
