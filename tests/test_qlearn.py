"""Tests for the Q-learning agent and self-play training.

Formal "stops losing to minimax" validation is Step 8; here we check the mechanics and a
basic learning signal: after training on tic-tac-toe, the agent should beat a random
player easily and rarely lose.
"""

from games.base import P1
from games.tictactoe import TicTacToe
from agents.random_agent import RandomAgent
from agents.qlearn import QLearningAgent, self_play_train, evaluate


def ttt():
    return TicTacToe()


def test_choose_returns_a_legal_move():
    agent = QLearningAgent(seed=0)
    assert agent.choose(TicTacToe()) in range(9)


def test_update_moves_value_toward_the_target():
    agent = QLearningAgent(alpha=0.5)
    key = TicTacToe().state_key()
    assert agent.q.get(key, {}).get(4, 0.0) == 0.0
    agent.update(key, 4, target=1.0)          # halfway to 1.0 with alpha=0.5
    assert agent.q[key][4] == 0.5
    agent.update(key, 4, target=1.0)          # halfway again -> 0.75
    assert agent.q[key][4] == 0.75


def test_greedy_choice_prefers_the_higher_valued_move():
    agent = QLearningAgent(epsilon=0.0, seed=1)
    game = TicTacToe()
    key = game.state_key()
    agent.update(key, 6, target=0.9)          # tell it move 6 is great
    assert agent.choose(game) == 6


def test_training_beats_a_random_opponent():
    agent = QLearningAgent(alpha=0.2, epsilon=0.2, seed=0)
    self_play_train(agent, ttt, games=15000)
    wins, draws, losses = evaluate(agent, RandomAgent(seed=999), ttt, games=200)
    # A tic-tac-toe agent trained this long should dominate random: far more wins than
    # losses, and losses should be a small fraction.
    assert wins > losses
    assert losses < 0.15 * (wins + draws + losses)


def test_learning_actually_helps_versus_no_training():
    untrained = QLearningAgent(epsilon=0.0, seed=0)
    trained = QLearningAgent(alpha=0.2, epsilon=0.2, seed=0)
    self_play_train(trained, ttt, games=15000)
    _, _, l_untrained = evaluate(untrained, RandomAgent(seed=7), ttt, games=200)
    _, _, l_trained = evaluate(trained, RandomAgent(seed=7), ttt, games=200)
    assert l_trained < l_untrained            # training reduces losses
