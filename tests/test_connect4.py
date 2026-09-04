"""Hand-checked correctness tests for the Connect-4 board.

Win-detection bugs are silent - they don't crash, they just report the wrong winner, and
then every downstream result is built on a lie. So we hammer the board with positions
where we already know the answer by eye. This is the plan's "highest-leverage hour".

To keep every test position hand-checkable, `make(...)` builds a board from an ASCII
picture, one string per row, top row first:

    "...."     '.' = empty
    "...."     'X' = player 1
    "OOO."     'O' = player 2
    "XXXX"

The pictures are drawn gravity-valid (no floating discs) so they're realistic, but note
that `winner()` is being tested in isolation and doesn't care how a position was reached.
"""

from games.base import EMPTY, P1, P2
from games.connect4 import Connect4

_CHAR = {".": EMPTY, "X": P1, "O": P2}


def make(grid, to_move=P1, connect=4):
    """Build a Connect4 from a list of equal-length strings (top row first)."""
    rows = len(grid)
    cols = len(grid[0])
    assert all(len(line) == cols for line in grid), "grid rows must be equal length"
    flat = tuple(_CHAR[ch] for line in grid for ch in line)
    return Connect4(rows, cols, connect, board=flat, to_move=to_move)


def cell(game, r, c):
    """Read cell (r, c) via the public state, without poking at internals."""
    board, _ = game.state_key()
    return board[r * game.cols + c]


# --- gravity & moves ---------------------------------------------------------

def test_dropped_disc_lands_at_the_bottom():
    g = Connect4(5, 4).apply(2)        # drop one disc in column 2
    assert cell(g, 4, 2) == P1         # it rests on the bottom row (row 4)
    # every other cell is still empty
    assert all(cell(g, r, c) == EMPTY
               for r in range(5) for c in range(4) if (r, c) != (4, 2))


def test_discs_stack_in_the_same_column():
    g = Connect4(5, 4)
    g = g.apply(1)                     # P1 lands on the floor (row 4)
    g = g.apply(1)                     # P2 stacks on top (row 3)
    assert cell(g, 4, 1) == P1
    assert cell(g, 3, 1) == P2


def test_full_column_leaves_legal_moves_and_raises():
    g = Connect4(5, 4)
    for _ in range(5):                 # 5 rows -> 5 discs fill the column
        g = g.apply(0)
    assert 0 not in g.legal_moves()    # the full column is no longer playable
    try:
        g.apply(0)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError when dropping into a full column"


def test_apply_does_not_mutate_the_original():
    start = Connect4(5, 4)
    start.apply(2)                     # throw away the result
    assert start.legal_moves() == [0, 1, 2, 3]
    assert all(cell(start, r, c) == EMPTY for r in range(5) for c in range(4))


def test_board_dimensions_are_parametric():
    assert Connect4(5, 4).legal_moves() == [0, 1, 2, 3]
    assert Connect4(5, 5).legal_moves() == [0, 1, 2, 3, 4]


# --- one clean win in every direction ---------------------------------------

def test_horizontal_win():
    g = make([
        "....",
        "....",
        "....",
        "OOO.",
        "XXXX",
    ])
    assert g.winner() == P1
    assert g.is_terminal()
    assert g.legal_moves() == []       # a finished game offers no moves


def test_vertical_win():
    g = make([
        "X...",
        "X...",
        "X...",
        "X...",
        "O.O.",
    ])
    assert g.winner() == P1


def test_diagonal_down_right_win():
    #  the '\' diagonal: (1,0) (2,1) (3,2) (4,3)
    g = make([
        "....",
        "X...",
        "OX..",
        "OOX.",
        "OOOX",
    ])
    assert g.winner() == P1


def test_diagonal_down_left_win():
    #  the '/' diagonal: (1,3) (2,2) (3,1) (4,0)
    g = make([
        "....",
        "...X",
        "..XO",
        ".XOO",
        "XOOO",
    ])
    assert g.winner() == P1


# --- the bug-catchers --------------------------------------------------------

def test_three_in_a_row_is_not_a_win():
    g = make([
        "....",
        "....",
        "....",
        "....",
        "XXX.",
    ])
    assert g.winner() is None          # three is not four
    assert not g.is_terminal()


def test_broken_line_is_not_a_win():
    # Four X's exist in the bottom row, but not four CONSECUTIVE (an O splits them).
    g = make([
        ".....",
        ".....",
        ".....",
        ".....",
        "XXXOX",
    ])
    assert g.winner() is None


def test_run_off_the_right_edge_is_not_a_win_and_does_not_crash():
    # Three X's against the right wall; a fourth would need a cell off the board.
    # This exercises the bounds check - a buggy scanner would read past the edge.
    g = make([
        ".....",
        ".....",
        ".....",
        ".....",
        "..XXX",
    ])
    assert g.winner() is None


# --- draw --------------------------------------------------------------------

def test_full_small_board_is_a_draw():
    # A 2x2 board can never fit four-in-a-row, so filling it is guaranteed a draw.
    g = Connect4(2, 2, connect=4)
    for col in [0, 0, 1, 1]:
        g = g.apply(col)
    assert g.winner() is None
    assert g.is_terminal()
    assert g.is_draw()
    assert g.legal_moves() == []


# --- state key ---------------------------------------------------------------

def test_state_key_distinguishes_whose_turn_it_is():
    same_board = (EMPTY,) * 20
    a = Connect4(5, 4, board=same_board, to_move=P1)
    b = Connect4(5, 4, board=same_board, to_move=P2)
    assert a.state_key() != b.state_key()   # identical cells, different player to move
