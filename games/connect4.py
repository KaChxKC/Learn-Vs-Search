"""Connect-4 with gravity - the real game, sized to order.

One class covers both project boards: `Connect4(5, 4)` and `Connect4(5, 5)`. Making it
parametric (rows, cols, and how many in a row you need) means we validate on the small
5x4 board and scale to 5x5 without touching this code at all.

It obeys the exact same `Game` interface as tic-tac-toe, so every agent we build later
runs on it unchanged. The three things that differ from tic-tac-toe:

  1. A MOVE IS A COLUMN, not a cell. You choose a column; gravity decides the row.
  2. The board size is variable, so win-detection can't use a hard-coded list of lines
     the way tic-tac-toe did - it has to scan generically.
  3. Wins can run in four directions on a bigger grid.

Coordinate convention (fixed once, used everywhere):

    row 0     is the TOP of the board
    row rows-1 is the BOTTOM (where discs come to rest)
    cell (r, c) lives at flat index  r * cols + c

    col:  0   1   2   3
        | . | . | . | . |   row 0  (top)
        | . | . | . | . |   row 1
        | . | . | . | . |   row 2
        | . | . | . | . |   row 3
        | . | . | X | . |   row 4  (bottom)  <- a disc dropped in column 2 rests here
"""

from games.base import Game, EMPTY, P1, P2, other, symbol

# The four directions a win can run. We scan every cell in each of these directions;
# scanning down-and-right and down-and-left (plus straight right and straight down)
# from every cell covers every possible line exactly once - no need for "up" or "left".
_DIRECTIONS = (
    (0, 1),    # horizontal  ->
    (1, 0),    # vertical    v
    (1, 1),    # diagonal    down-right  \
    (1, -1),   # diagonal    down-left   /
)


class Connect4(Game):
    def __init__(self, rows, cols, connect=4, board=None, to_move=P1):
        self.rows = rows
        self.cols = cols
        self.connect = connect
        # The board is an immutable flat tuple of rows*cols cells.
        self._board = board if board is not None else (EMPTY,) * (rows * cols)
        self._to_move = to_move

    # --- small helpers -------------------------------------------------------

    def _at(self, r, c):
        """Value of cell (r, c). Assumes r, c are in bounds."""
        return self._board[r * self.cols + c]

    def _landing_row(self, col):
        """The row a disc dropped in `col` would rest in, or None if the column is full.

        Gravity: scan from the BOTTOM row upward and return the first empty slot found.
        """
        for r in range(self.rows - 1, -1, -1):   # rows-1, rows-2, ..., 0
            if self._at(r, col) == EMPTY:
                return r
        return None   # column is full

    # --- the Game interface --------------------------------------------------

    def legal_moves(self):
        # No moves once someone has won.
        if self.winner() is not None:
            return []
        # A column is playable if its TOP cell (row 0) is still empty.
        return [c for c in range(self.cols) if self._at(0, c) == EMPTY]

    def apply(self, move):
        col = move
        row = self._landing_row(col)
        if row is None:
            raise ValueError(f"column {col} is full")
        new_board = list(self._board)
        new_board[row * self.cols + col] = self._to_move
        return Connect4(self.rows, self.cols, self.connect,
                        tuple(new_board), other(self._to_move))

    def winner(self):
        for r in range(self.rows):
            for c in range(self.cols):
                player = self._at(r, c)
                if player == EMPTY:
                    continue
                # From this cell, does a run of `connect` same-player discs start
                # in any of the four directions?
                for dr, dc in _DIRECTIONS:
                    if self._run_length(r, c, dr, dc, player):
                        return player
        return None

    def _run_length(self, r, c, dr, dc, player):
        """True if `connect` cells starting at (r, c) heading (dr, dc) are all `player`."""
        end_r = r + (self.connect - 1) * dr
        end_c = c + (self.connect - 1) * dc
        # Bail early if the far end of the line would fall off the board.
        if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
            return False
        for step in range(self.connect):
            if self._at(r + step * dr, c + step * dc) != player:
                return False
        return True

    def current_player(self):
        return self._to_move

    def state_key(self):
        return (self._board, self._to_move)

    def __str__(self):
        header = "  " + "   ".join(str(c) for c in range(self.cols))
        lines = [header]
        for r in range(self.rows):
            cells = (symbol(self._at(r, c)) for c in range(self.cols))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)
