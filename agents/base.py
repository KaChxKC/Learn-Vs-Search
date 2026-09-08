"""The one interface every player must satisfy.

This is the mirror image of the `Game` interface. A board answers questions about a
position; an agent looks at a position and picks a move. The whole project talks through
these two tiny contracts, which is why a random player, a minimax searcher, and a
Q-learning agent are completely interchangeable - and why the UI can be a dumb viewer
that just calls `choose` without knowing what kind of brain is behind it.
"""

from abc import ABC, abstractmethod


class Agent(ABC):
    #: A short human-readable label, handy for printing who is playing.
    name = "agent"

    @abstractmethod
    def choose(self, game):
        """Return ONE legal move for the player whose turn it is in `game`.

        The move must be one of `game.legal_moves()`. `game` is never modified - an
        agent only inspects it and returns a move; the caller applies it.
        """
