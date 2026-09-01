"""Tic-tac-toe (3x3) - the proof-of-life board.

It is a solved game, so it has known-correct answers: full-depth minimax must never lose,
and a trained Q-agent must stop losing against it. That makes it the cheapest place to
catch bugs in the search and (especially) the two-player Q-learning credit assignment,
before we ever scale up to real Connect-4.

A move here is a cell index 0..8, laid out like:

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

from games.base import Game, EMPTY, P1, P2, other, symbol

SIZE = 3

# The eight winning lines, as triples of cell indices.
_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
)


class TicTacToe(Game):
    def __init__(self, board=None, to_move=P1):
        # board is an immutable tuple of 9 cells.
        self._board = board if board is not None else (EMPTY,) * (SIZE * SIZE)
        self._to_move = to_move

    def legal_moves(self):
        # No moves once the game is already won.
        if self.winner() is not None:
            return []
        return [i for i, cell in enumerate(self._board) if cell == EMPTY]

    def apply(self, move):
        if self._board[move] != EMPTY:
            raise ValueError(f"cell {move} is already taken")
        new_board = list(self._board)
        new_board[move] = self._to_move
        return TicTacToe(tuple(new_board), other(self._to_move))

    def winner(self):
        for a, b, c in _LINES:
            if self._board[a] != EMPTY and self._board[a] == self._board[b] == self._board[c]:
                return self._board[a]
        return None

    def current_player(self):
        return self._to_move

    def state_key(self):
        # The board plus whose turn it is uniquely identifies the position.
        return (self._board, self._to_move)

    def __str__(self):
        rows = []
        for r in range(SIZE):
            cells = (symbol(self._board[r * SIZE + c]) for c in range(SIZE))
            rows.append(" " + " | ".join(cells))
        return ("\n" + "-" * 11 + "\n").join(rows)
