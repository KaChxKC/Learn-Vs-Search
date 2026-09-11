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

import random

from agents.base import Agent
from games.base import P1, P2


class QLearningAgent(Agent):
    name = "qlearn"

    def __init__(self, alpha=0.1, gamma=1.0, epsilon=0.1,
                 draw_reward=0.0, seed=None, q=None):
        self.alpha = alpha              # learning rate: how far to nudge toward the target
        self.gamma = gamma              # discount: ~1.0, these games are short and episodic
        self.epsilon = epsilon          # exploration rate used only when explore=True
        self.draw_reward = draw_reward  # value of a draw (tunable later; 0 = neutral)
        self.q = q if q is not None else {}   # Q[state_key] -> {move: value}
        self._rng = random.Random(seed)

    # --- reading the table ---------------------------------------------------

    def _row(self, state_key):
        row = self.q.get(state_key)
        if row is None:
            row = {}
            self.q[state_key] = row
        return row

    def value(self, game):
        """Best Q available to the player to move in `game` (0 if nothing is known)."""
        row = self.q.get(game.state_key(), {})
        return max((row.get(m, 0.0) for m in game.legal_moves()), default=0.0)

    # --- choosing a move -----------------------------------------------------

    def choose(self, game, explore=False):
        legal = game.legal_moves()
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(legal)          # explore
        row = self.q.get(game.state_key(), {})
        best = max(row.get(m, 0.0) for m in legal)
        best_moves = [m for m in legal if row.get(m, 0.0) == best]
        return self._rng.choice(best_moves)         # exploit (random tie-break)

    # --- learning ------------------------------------------------------------

    def update(self, state_key, move, target):
        row = self._row(state_key)
        old = row.get(move, 0.0)
        row[move] = old + self.alpha * (target - old)


def self_play_train(agent, make_game, games):
    """Train `agent` by having it play `games` complete games against itself.

    One agent plays both seats and updates after every move, so BOTH sides' positions
    get learned automatically - which is why pure self-play needs no seat-alternation.
    """
    for _ in range(games):
        game = make_game()
        while not game.is_terminal():
            state_key = game.state_key()
            move = agent.choose(game, explore=True)
            nxt = game.apply(move)

            if nxt.is_terminal():
                winner = nxt.winner()
                # The mover either won or drew (you can't hand the opponent a win by moving).
                target = agent.draw_reward if winner is None else 1.0
            else:
                # Minus: the opponent's best outcome in nxt is our worst.
                target = agent.gamma * (-agent.value(nxt))

            agent.update(state_key, move, target)
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
