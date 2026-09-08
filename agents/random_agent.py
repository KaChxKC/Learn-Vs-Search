"""A player that picks a legal move uniformly at random.

Why bother with something this dumb? Two reasons:
  1. It is the baseline. "Beats a random player convincingly" is the first sanity check
     a trained Q-agent must pass (Step 9) before we trust any fancier result.
  2. It exercises the whole Agent/Game plumbing end-to-end before we add real thinking,
     so if something is wired wrong we find out now, cheaply.

It takes an optional seed so games can be reproduced exactly - the same seed always
produces the same choices, which makes tests and debugging deterministic.
"""

import random

from agents.base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed=None):
        # A private random generator, so this agent's randomness is independent of
        # (and doesn't disturb) anything else using the global `random` module.
        self._rng = random.Random(seed)

    def choose(self, game):
        return self._rng.choice(game.legal_moves())
