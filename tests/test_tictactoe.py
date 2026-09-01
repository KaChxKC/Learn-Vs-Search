"""Hand-checked correctness tests for the tic-tac-toe board.

Win detection is the classic silent killer, so we nail it here on the small board where
every case is easy to reason about by eye.
"""

from games.base import P1, P2
from games.tictactoe import TicTacToe


def play(moves):
    """Apply a sequence of moves starting from an empty board."""
    game = TicTacToe()
    for m in moves:
        game = game.apply(m)
    return game


def test_new_board_has_nine_moves_and_p1_starts():
    game = TicTacToe()
    assert game.current_player() == P1
    assert sorted(game.legal_moves()) == list(range(9))
    assert game.winner() is None
    assert not game.is_terminal()


def test_apply_advances_player_and_consumes_the_cell():
    game = TicTacToe().apply(4)
    assert game.current_player() == P2
    assert 4 not in game.legal_moves()
    assert len(game.legal_moves()) == 8


def test_apply_does_not_mutate_the_original():
    start = TicTacToe()
    start.apply(0)
    # Original is untouched: games are immutable.
    assert sorted(start.legal_moves()) == list(range(9))


def test_apply_to_taken_cell_raises():
    game = TicTacToe().apply(0)
    try:
        game.apply(0)
    except ValueError:
        return
    assert False, "expected ValueError when playing a taken cell"


def test_row_win():
    # X: 0,1,2   O: 3,4
    game = play([0, 3, 1, 4, 2])
    assert game.winner() == P1
    assert game.is_terminal()
    assert game.legal_moves() == []


def test_column_win():
    # X: 0,3,6   O: 1,2
    game = play([0, 1, 3, 2, 6])
    assert game.winner() == P1


def test_diagonal_win():
    # X: 0,4,8   O: 1,2
    game = play([0, 1, 4, 2, 8])
    assert game.winner() == P1


def test_anti_diagonal_win():
    # X: 2,4,6   O: 0,1
    game = play([2, 0, 4, 1, 6])
    assert game.winner() == P1


def test_full_board_draw():
    # A known drawn fill:
    #  X | O | X
    #  X | O | O
    #  O | X | X
    game = play([0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert game.winner() is None
    assert game.is_terminal()
    assert game.is_draw()


def test_near_miss_is_not_a_win():
    # Two in a row is not three: must not false-positive.
    game = play([0, 3, 1])  # X at 0,1 only
    assert game.winner() is None
    assert not game.is_terminal()
