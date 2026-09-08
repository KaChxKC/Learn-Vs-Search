"""Minimax, with optional alpha-beta pruning.

WHAT MINIMAX DOES (recap)
-------------------------
It scores every position from the perspective of the player about to move (+1 win, 0 draw,
-1 loss) by playing every line to the end and backing the results up the tree. The key
line is `value = -child_value`: my opponent's gain is my loss, so one recursive function
(negamax) handles both players.

WHAT ALPHA-BETA ADDS
--------------------
Plain minimax explores the ENTIRE tree, including branches it should already know are
pointless. Alpha-beta carries two bounds down the search:

    alpha = the best score the side to move is already guaranteed elsewhere
    beta  = the best score the opponent is already guaranteed elsewhere

The moment a branch proves it can't beat what the opponent already has (`alpha >= beta`),
we STOP searching it - any deeper look could only confirm it's too good for the opponent
to ever allow. This is the "cutoff". It never changes which move is chosen (a branch is
only abandoned once it's proven irrelevant), it just skips work.

We keep a `prune` switch so we can run BOTH versions and compare them directly:
identical moves, dramatically fewer nodes. `self.nodes` counts positions examined in the
last `choose()` - that count is the data behind the alpha-beta efficiency graph.
"""

from agents.base import Agent

_INF = float("inf")


class MinimaxAgent(Agent):
    def __init__(self, depth=None, prune=True):
        self.depth = depth          # None = search to the end of the game
        self.prune = prune          # True = alpha-beta; False = plain minimax
        kind = "alphabeta" if prune else "minimax"
        self.name = kind if depth is None else f"{kind}(d={depth})"
        self.nodes = 0              # positions examined in the last choose()

    def choose(self, game):
        self.nodes = 0
        alpha, beta = -_INF, _INF
        best_value, best_move = -_INF, None
        for move in game.legal_moves():
            child = game.apply(move)
            value = -self._negamax(child, self._step(self.depth), -beta, -alpha)
            if value > best_value:          # strict '>' keeps tie-breaking identical
                best_value, best_move = value, move
            if best_value > alpha:          # tightening alpha lets children prune more
                alpha = best_value
        return best_move

    def _negamax(self, game, depth, alpha, beta):
        """Best value the player to move can force in `game`, within the (alpha, beta) window."""
        self.nodes += 1

        if game.is_terminal():
            if game.winner() is None:
                return 0
            return -1                        # someone just won -> player to move has lost

        if depth == 0:
            return self._heuristic(game)

        best = -_INF
        for move in game.legal_moves():
            child = game.apply(move)
            # Search the child with the window flipped and negated - the opponent's
            # alpha/beta are our beta/alpha, mirrored.
            value = -self._negamax(child, self._step(depth), -beta, -alpha)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if self.prune and alpha >= beta:
                break                        # cutoff: this branch can't matter
        return best

    @staticmethod
    def _step(depth):
        """Count one level deeper. None stays None (unlimited); an int counts down."""
        return None if depth is None else depth - 1

    @staticmethod
    def _heuristic(game):
        """Score a non-terminal position we couldn't search to the end. Neutral for now."""
        return 0
