"""A player that asks YOU for the move, by reading from the terminal.

This is the heart of the plan's "scrappy terminal play mode" - a debugging tool, not a
deliverable. Playing by hand against your agents surfaces blunders that raw win-rate
numbers hide. It keeps asking until you type a legal move, so a typo never crashes a game.
"""

from agents.base import Agent


class HumanAgent(Agent):
    name = "human"

    def choose(self, game):
        legal = game.legal_moves()
        while True:
            raw = input(f"  your move {legal} > ").strip()
            try:
                move = int(raw)
            except ValueError:
                print(f"  '{raw}' is not a number - try again")
                continue
            if move in legal:
                return move
            print(f"  {move} is not a legal move; choose from {legal}")
