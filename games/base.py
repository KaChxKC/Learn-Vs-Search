"""The one interface every board must satisfy.

An agent only ever talks to a board through this `Game` interface, so minimax and
Q-learning are written once and work on tic-tac-toe, 5x4 Connect-4, and 5x5 Connect-4
without changing a line. Keeping this interface small is what makes that possible.

Conventions used everywhere in the project:
  * Cells hold EMPTY, P1, or P2 (0, 1, 2).
  * A "move" is always a single integer (a cell index for tic-tac-toe, a column for
    Connect-4). So `legal_moves()` returns a list of ints for every board.
  * Games are immutable: `apply(move)` returns a NEW game rather than mutating this one.
    That removes a whole class of make/unmake bugs from the search and learning code.
"""

from abc import ABC, abstractmethod

EMPTY, P1, P2 = 0, 1, 2


def other(player):
    """The opponent of `player`."""
    return P2 if player == P1 else P1


def symbol(cell):
    """Human-readable single character for a cell value (for terminal display)."""
    return {EMPTY: ".", P1: "X", P2: "O"}[cell]


class Game(ABC):
    """Abstract board. See module docstring for the conventions."""

    @abstractmethod
    def legal_moves(self):
        """List of integer moves that are currently legal. Empty if the game is over."""

    @abstractmethod
    def apply(self, move):
        """Return a NEW Game with `move` played by the current player."""

    @abstractmethod
    def winner(self):
        """P1 or P2 if that player has won, else None (includes an ongoing game or a draw)."""

    @abstractmethod
    def current_player(self):
        """Whose turn it is: P1 or P2."""

    @abstractmethod
    def state_key(self):
        """A small hashable key identifying this position. Used as the Q-table key."""

    def is_terminal(self):
        """True when the game is over: someone won, or there are no legal moves left."""
        return self.winner() is not None or not self.legal_moves()

    def is_draw(self):
        """True when the game is over with no winner."""
        return self.is_terminal() and self.winner() is None
