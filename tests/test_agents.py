"""Tests for the agents and the play plumbing.

The human agent needs keyboard input, so we don't unit-test it here; it's exercised by
hand via play.py. The random agent and the game-driving loop we can test fully.
"""

from games.base import P1, P2
from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from play import play


def test_random_agent_only_ever_returns_legal_moves():
    agent = RandomAgent(seed=0)
    game = Connect4(5, 4)
    while not game.is_terminal():
        move = agent.choose(game)
        assert move in game.legal_moves()   # never picks an illegal move
        game = game.apply(move)


def test_random_agent_is_reproducible_with_a_seed():
    game = TicTacToe()
    # Two agents with the same seed, asked about the same position, must agree.
    a = RandomAgent(seed=42)
    b = RandomAgent(seed=42)
    picks_a = []
    picks_b = []
    g = game
    for _ in range(5):
        picks_a.append(a.choose(g))
        picks_b.append(b.choose(g))
        g = g.apply(picks_a[-1])
    assert picks_a == picks_b


def test_different_seeds_can_differ():
    # Not a guarantee for a single call, but across a whole game two different seeds
    # should almost never produce identical move sequences. We just check the API runs.
    a = RandomAgent(seed=1)
    b = RandomAgent(seed=2)
    assert a.choose(TicTacToe()) in range(9)
    assert b.choose(TicTacToe()) in range(9)


def test_random_vs_random_game_always_terminates_cleanly():
    # Drive a full game through the same loop play.py uses and confirm it ends in a
    # valid terminal state without ever making an illegal move (apply would raise).
    game = Connect4(5, 5)
    players = {P1: RandomAgent(seed=1), P2: RandomAgent(seed=2)}
    steps = 0
    while not game.is_terminal():
        move = players[game.current_player()].choose(game)
        game = game.apply(move)
        steps += 1
        assert steps <= 5 * 5 + 1     # can't exceed one move per cell
    assert game.winner() in (P1, P2, None)


def test_play_returns_a_valid_result():
    # The play() driver returns the winner (or None for a draw).
    result = play(TicTacToe(), RandomAgent(seed=3), RandomAgent(seed=4))
    assert result in (P1, P2, None)
