"""Minimax - the first agent that actually thinks ahead.

THE IDEA
--------
Minimax assumes both players play perfectly. To score a position for the player whose
turn it is, it asks: "for each move I could make, my opponent will then reply with THEIR
best move, and so on, all the way to the end of the game." It plays out every line to a
finish (win / lose / draw), then backs those results up the tree and plays the move that
leads to the best guaranteed outcome.

THE TRICK WE USE (negamax)
--------------------------
Instead of writing separate "maximise for X / minimise for O" code, we score every
position FROM THE PERSPECTIVE OF THE PLAYER ABOUT TO MOVE, using just three values:

    +1  this player can force a win
     0  best play leads to a draw
    -1  this player will lose to best play

The key line is `value = -best_child_value`: my opponent's good news is my bad news, so
whatever the position is worth to them after my move, it's worth the NEGATIVE of that to
me. One recursive function handles both players. That symmetry is why it's called negamax.

DEPTH LIMIT
-----------
`depth=None` searches all the way to the end of the game - perfect for a small solved
game like tic-tac-toe. For big boards the full tree is astronomically large, so we pass a
`depth` (how many moves to look ahead); when we run out of depth at a non-terminal
position we fall back to a heuristic guess. For now that heuristic is a neutral 0 ("I
can't see far enough to tell"), which we can improve later if a depth needs to play
stronger.
"""

from agents.base import Agent

_INF = float("inf")


class MinimaxAgent(Agent):
    def __init__(self, depth=None):
        self.depth = depth                      # None = search to the end of the game
        self.name = "minimax" if depth is None else f"minimax(d={depth})"
        self.nodes = 0                          # positions examined in the last choose()

    def choose(self, game):
        self.nodes = 0
        best_value = -_INF
        best_move = None
        for move in game.legal_moves():
            # Value of this move = the negation of what the resulting position is worth
            # to the OPPONENT (who moves next in the child position).
            child = game.apply(move)
            value = -self._negamax(child, self._step(self.depth))
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def _negamax(self, game, depth):
        """How good is `game` for the player whose turn it is now? (+1 / 0 / -1)."""
        self.nodes += 1

        # Base case 1: the game is over - a concrete, known answer.
        if game.is_terminal():
            if game.winner() is None:
                return 0                        # a draw is worth nothing to either side
            # If there's a winner at a terminal node, it's the player who just moved -
            # so the player about to move here has already LOST.
            return -1

        # Base case 2: we've looked as far ahead as we're allowed - guess.
        if depth == 0:
            return self._heuristic(game)

        # Recursive case: try every move, keep the best value we can force.
        best = -_INF
        for move in game.legal_moves():
            child = game.apply(move)
            best = max(best, -self._negamax(child, self._step(depth)))
        return best

    @staticmethod
    def _step(depth):
        """Count one level deeper. None stays None (unlimited); an int counts down."""
        return None if depth is None else depth - 1

    @staticmethod
    def _heuristic(game):
        """Score a non-terminal position we couldn't search to the end. Neutral for now."""
        return 0
