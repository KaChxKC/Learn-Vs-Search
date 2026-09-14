"""Step 9: the learner on the REAL Connect-4 mechanics (5x4).

Nothing new is built for this step - and that's the whole point. Because every board
obeys the same Game interface, the Q-agent, self-play training, and evaluation all work
on 5x4 Connect-4 (gravity, four-in-a-row, diagonals, full columns) with ZERO code
changes. This test just confirms the learner actually gets good on the real game, not
only on tic-tac-toe.

It trains for a while, so it takes several seconds.
"""

from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.qlearn import QLearningAgent, self_play_train, evaluate


def c5x4():
    return Connect4(5, 4)


def test_learner_beats_random_on_real_connect4_5x4():
    trained = QLearningAgent(alpha=0.2, epsilon=0.2, seed=0)
    self_play_train(trained, c5x4, games=50000)
    untrained = QLearningAgent(epsilon=0.0, seed=0)

    tw, td, tl = evaluate(trained, RandomAgent(seed=7), c5x4, games=200)
    uw, ud, ul = evaluate(untrained, RandomAgent(seed=7), c5x4, games=200)

    # Honest bar for 5x4 at this scale: it beats random (more wins than losses) and
    # training clearly helped. It does NOT yet crush random - the state space is large
    # and coverage is thin until symmetry folding + longer runs (Step 10). See the
    # ~46% win rate on a 200k-game run for why the bar is set here, not higher.
    assert tw > tl         # beats random: wins more than it loses
    assert tw > uw         # training clearly helped versus an untrained table
