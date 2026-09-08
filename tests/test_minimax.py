"""Tests for the minimax agent.

Tic-tac-toe is a SOLVED game, so full-depth minimax has known-correct behaviour:
  * it must never lose, from either seat;
  * two perfect players must always draw.
A failure here is an unambiguous bug signal, which is exactly why we validate on the
small solved board before trusting minimax as the yardstick for the learning agent.
"""

from games.base import P1, P2, EMPTY
from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.minimax import MinimaxAgent


def _play(game, players):
    while not game.is_terminal():
        game = game.apply(players[game.current_player()].choose(game))
    return game.winner()


def test_minimax_never_loses_to_random_from_either_seat():
    # Play many games; minimax must never come out the loser, whether it moves first
    # (as X) or second (as O). Kept to a handful of seeds because PLAIN minimax has no
    # pruning and is slow - Step 6 (alpha-beta) makes larger sweeps cheap.
    for seed in range(8):
        # minimax as X
        w = _play(TicTacToe(), {P1: MinimaxAgent(), P2: RandomAgent(seed=seed)})
        assert w != P2, f"minimax (X) lost with seed {seed}"
        # minimax as O
        w = _play(TicTacToe(), {P1: RandomAgent(seed=seed), P2: MinimaxAgent()})
        assert w != P1, f"minimax (O) lost with seed {seed}"


def test_two_perfect_players_always_draw():
    w = _play(TicTacToe(), {P1: MinimaxAgent(), P2: MinimaxAgent()})
    assert w is None


def test_minimax_takes_an_immediate_win():
    # X has two in the top row (cells 0,1); cell 2 completes it. X to move.
    #  X | X | .
    #  O | O | .
    #  . | . | .
    board = (P1, P1, EMPTY, P2, P2, EMPTY, EMPTY, EMPTY, EMPTY)
    game = TicTacToe(board=board, to_move=P1)
    assert MinimaxAgent().choose(game) == 2


def test_minimax_blocks_an_immediate_loss():
    # O threatens the top row (cells 0,1); X must block at 2 or lose. X to move.
    #  O | O | .
    #  . | X | .
    #  . | . | X
    board = (P2, P2, EMPTY, EMPTY, P1, EMPTY, EMPTY, EMPTY, P1)
    game = TicTacToe(board=board, to_move=P1)
    assert MinimaxAgent().choose(game) == 2


def test_depth_limited_minimax_still_sees_an_immediate_connect4_win():
    # Bottom row: X X X . with O O O stacked above. X to move; dropping column 3 wins.
    game = Connect4(5, 4)
    for m in [0, 0, 1, 1, 2, 2]:
        game = game.apply(m)
    assert MinimaxAgent(depth=1).choose(game) == 3


def test_minimax_returns_a_legal_move_on_a_big_board():
    game = Connect4(5, 5)
    move = MinimaxAgent(depth=2).choose(game)
    assert move in game.legal_moves()
