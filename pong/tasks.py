"""
Invoke task runner for the ai-pong project.

Requires: pip install invoke

Usage examples:
    invoke test                     # run the full pytest suite
    invoke test.qlearning           # q-learning tests only
    invoke test.unit                # legacy physics/collision tests only
    invoke test.smoke               # headless game smoke run (log renderer)

    invoke train.qlearning --episodes 100 --epsilon 0.8
    invoke train.selfplay --episodes 200 --opponent-epsilon 0.1
    invoke train.perfect --episodes 100

    invoke play --left perfect --right qlearning
    invoke clean
"""

from invoke import task, Collection


# ----------------------------------------------------------------------
# Testing
# ----------------------------------------------------------------------
@task
def unit(c):
    """Run the full pytest suite."""
    c.run("pytest -q --tb=short")


@task
def qlearning(c):
    """Run only the Q-learning test module."""
    c.run("pytest -q --tb=short test_qlearning.py")


@task
def legacy(c):
    """Run the legacy test modules (physics, collision, wall bounce, ...)."""
    c.run("pytest -q --tb=short --ignore=test_qlearning.py")


@task(help={"frames": "Number of headless frames (default: 300)"})
def smoke(c, frames: int = 300):
    """Headless smoke run of the game with the log renderer."""
    c.run(f"python pong.py --renderer log --fast {frames}")


test = Collection("test")
test.add_task(unit, "unit")
test.add_task(qlearning, "qlearning")
test.add_task(legacy, "legacy")
test.add_task(smoke, "smoke")


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
@task(help={
    "episodes": "Number of episodes (default: 100)",
    "epsilon": "Initial exploration rate 0-1 (default: 0.8)",
    "opponent": "Opponent: stationary/algorithmic/perfect (default: stationary)",
    "interval": "Report bounces/W/L every N episodes (default: 25)",
    "save": "Save the trained agent to this path (default: trained_agent.pkl, "
            "empty string disables saving)",
})
def qlearning_(c, episodes: int = 100, epsilon: float = 0.8,
               opponent: str = "stationary", interval: int = 25,
               save: str = "trained_agent.pkl"):
    """Train against an opponent, reporting bounces vs W/L per interval
    (train_qlearning.py). Watch the 'bounces/ep' column climb as the agent
    learns while 'L' stays flat or drops - that is the improvement signal."""
    c.run(
        f"python train_qlearning.py --episodes {episodes} --epsilon {epsilon} "
        f"--opponent {opponent} --interval {interval} --save {save}",
        echo=True,
    )


@task(help={
    "episodes": "Number of episodes (default: 200)",
    "opponent_epsilon": "Opponent exploration rate (default: 0.1)",
})
def selfplay(c, episodes: int = 200, opponent_epsilon: float = 0.1):
    """Train two agents against each other (train_agent.py)."""
    c.run(
        f"python train_agent.py --method selfplay --episodes {episodes} "
        f"--opponent-epsilon {opponent_epsilon}",
        echo=True,
    )


@task(help={"episodes": "Number of episodes (default: 100)"})
def perfect(c, episodes: int = 100):
    """Train against the perfect controller (train_agent.py)."""
    c.run(
        f"python train_agent.py --method perfect --episodes {episodes}",
        echo=True,
    )


train = Collection("train")
train.add_task(qlearning_, "qlearning")
train.add_task(selfplay, "selfplay")
train.add_task(perfect, "perfect")


# ----------------------------------------------------------------------
# Top-level tasks
# ----------------------------------------------------------------------
@task(help={
    "left": "Left controller: human/algorithmic/perfect/nn/qlearning "
            "(default: perfect)",
    "right": "Right controller (default: qlearning)",
    "agent": "Path to a saved Q-learning agent (default: trained_agent.pkl)",
})
def play(c, left: str = "perfect", right: str = "qlearning",
         agent: str = "trained_agent.pkl"):
    """Play the game interactively (TUI renderer). qlearning controllers use
    the saved agent, or play randomly with a warning if none exists."""
    c.run(f"python pong.py --left {left} --right {right} --agent {agent}")


@task
def clean(c):
    """Remove caches (__pycache__, .pytest_cache)."""
    c.run("find . -name '__pycache__' -type d -prune -exec rm -rf {} +")
    c.run("rm -rf .pytest_cache")


ns = Collection()
ns.add_collection(test)
ns.add_collection(train)
ns.add_task(play)
ns.add_task(clean)