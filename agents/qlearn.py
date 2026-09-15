"""Tabular Q-learning that learns purely from self-play.

THE TABLE
---------
The whole "brain" is a dictionary:  Q[state_key][move] = a number saying how good that
move looks from that position, for the player about to move. Unseen entries default to 0
("no opinion yet"). Tic-tac-toe fills a few thousand entries; 5x5 Connect-4 fills
hundreds of thousands - but it's still just a dict.

CHOOSING A MOVE (epsilon-greedy)
--------------------------------
While training we must balance using what we know against discovering new lines:
  * with probability epsilon -> play a random legal move (explore),
  * otherwise                -> play the best-known move (exploit).
When actually playing/evaluating we turn exploration off and always play greedily.

THE UPDATE (the two-player heart of the project)
------------------------------------------------
After the player to move in `s` plays `a`, we reach `s'`, where it's now the OPPONENT's
turn. We nudge Q[s][a] toward a target:

    if s' ends the game:   target = +1 (mover won)  or  draw_reward (drawn)
    otherwise:             target = gamma * ( - value(s') )

where value(s') is the opponent's best Q in s'. The MINUS is the crucial bit: this is a
zero-sum game, so whatever is good for the opponent in s' is exactly that bad for us.
Getting that sign right is the two-player credit assignment the plan warns about most.
"""

import gzip
import pickle
import random

from agents.base import Agent
from games.base import P1, P2


class QLearningAgent(Agent):
    name = "qlearn"

    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1,
                 draw_reward=0.0, seed=None, q=None, fold=False):
        self.alpha = alpha              # learning rate: how far to nudge toward the target
        self.gamma = gamma              # discount: ~1.0, these games are short and episodic
        self.epsilon = epsilon          # exploration rate used only when explore=True
        self.draw_reward = draw_reward  # value of a draw (tunable later; 0 = neutral)
        self.fold = fold                # fold mirror-image positions onto one key?
        self.q = q if q is not None else {}   # Q[state_key] -> {move: value}
        self._rng = random.Random(seed)

    # --- keys & rows ---------------------------------------------------------

    def _keymap(self, game):
        """The table key for this position, plus how to map real moves into that key.

        With folding on, mirror-image boards share one key and moves may be remapped
        (move_map). With folding off, it's just the raw state key and no remapping.
        """
        if self.fold:
            return game.canonical()
        return game.state_key(), None

    def _row(self, key):
        row = self.q.get(key)
        if row is None:
            row = {}
            self.q[key] = row
        return row

    def value(self, game):
        """Best Q available to the player to move in `game` (0 if nothing is known)."""
        key, move_map = self._keymap(game)
        row = self.q.get(key, {})
        legal = game.legal_moves()
        return max((row.get(move_map[m] if move_map else m, 0.0) for m in legal),
                   default=0.0)

    # --- choosing a move -----------------------------------------------------

    def choose(self, game, explore=False):
        legal = game.legal_moves()
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(legal)          # explore
        key, move_map = self._keymap(game)
        row = self.q.get(key, {})

        def q_of(move):                             # value of a real move via its key move
            return row.get(move_map[move] if move_map else move, 0.0)

        best = max(q_of(m) for m in legal)
        best_moves = [m for m in legal if q_of(m) == best]
        return self._rng.choice(best_moves)         # exploit (random tie-break)

    # --- learning ------------------------------------------------------------

    def learn(self, game, move, target):
        """Nudge the value of `move` (played in `game`) toward `target`, folding-aware."""
        key, move_map = self._keymap(game)
        key_move = move_map[move] if move_map else move
        self.update(key, key_move, target)

    def update(self, key, move, target):
        """Low-level table write (already-canonical key and move)."""
        row = self._row(key)
        old = row.get(move, 0.0)
        row[move] = old + self.alpha * (target - old)


def save_table(path, q, meta=None):
    """Save a Q-table (gzip-pickled) together with its metadata (e.g. whether folded)."""
    with gzip.open(path, "wb") as f:
        pickle.dump({"q": q, "meta": meta or {}}, f)


def load_table(path):
    """Load a table saved by save_table. Returns (q, meta).

    Also accepts an old raw-dict table (no metadata) for backward compatibility.
    """
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "q" in obj and "meta" in obj:
        return obj["q"], obj["meta"]
    return obj, {}          # legacy format: a bare Q-table, no metadata


def self_play_train(agent, make_game, games):
    """Train `agent` by having it play `games` complete games against itself.

    One agent plays both seats and updates after every move, so BOTH sides' positions
    get learned automatically - which is why pure self-play needs no seat-alternation.
    """
    for _ in range(games):
        game = make_game()
        while not game.is_terminal():
            move = agent.choose(game, explore=True)
            nxt = game.apply(move)

            if nxt.is_terminal():
                winner = nxt.winner()
                # The mover either won or drew (you can't hand the opponent a win by moving).
                target = agent.draw_reward if winner is None else 1.0
            else:
                # Minus: the opponent's best outcome in nxt is our worst.
                target = agent.gamma * (-agent.value(nxt))

            agent.learn(game, move, target)
            game = nxt


def evaluate(agent, opponent, make_game, games=400, seed=0):
    """Play `agent` vs `opponent`, alternating seats exactly 50/50. Greedy (no explore).

    Returns (wins, draws, losses) from the agent's point of view.
    """
    wins = draws = losses = 0
    for i in range(games):
        game = make_game()
        if i % 2 == 0:
            players, me = {P1: agent, P2: opponent}, P1
        else:
            players, me = {P1: opponent, P2: agent}, P2
        while not game.is_terminal():
            game = game.apply(players[game.current_player()].choose(game))
        winner = game.winner()
        if winner is None:
            draws += 1
        elif winner == me:
            wins += 1
        else:
            losses += 1
    return wins, draws, losses
