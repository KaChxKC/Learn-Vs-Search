"""Play a game in the terminal - the project's hands-on debugging tool.

Examples (run from the repo root):

    python play.py                          # you (X) vs a random agent (O), tic-tac-toe
    python play.py --board 5x4              # Connect-4 5x4 instead
    python play.py --board 5x5 --o human    # a two-human game on the 5x5 board
    python play.py --x random --o random    # watch two random agents play (no input)
    python play.py --x random --o random --seed 1   # ...reproducibly

This file has NO game rules and NO AI of its own. It just wires a board to two agents and
loops: show the board, ask the current player's agent for a move, apply it, repeat. That
is exactly the "viewer, never a player" shape the Pygame UI will take later.
"""

import argparse

from games import TicTacToe, Connect4, P1, P2, symbol
from agents import RandomAgent, HumanAgent, MinimaxAgent

# Each board is a zero-argument builder, so we only construct the one that's chosen.
BOARDS = {
    "ttt": lambda: TicTacToe(),
    "5x4": lambda: Connect4(5, 4),
    "5x5": lambda: Connect4(5, 5),
}

# Each agent builder takes the seed (random uses it; the others ignore it).
AGENTS = {
    "human": lambda seed: HumanAgent(),
    "random": lambda seed: RandomAgent(seed),
    "minimax": lambda seed: MinimaxAgent(),          # full-depth (great for tic-tac-toe)
    "minimax3": lambda seed: MinimaxAgent(depth=3),  # depth-limited (for the big boards)
}


def play(game, x_agent, o_agent):
    """Run one game to the end, printing as it goes. Returns the winner (P1/P2) or None."""
    players = {P1: x_agent, P2: o_agent}
    print(game)
    print()
    while not game.is_terminal():
        player = game.current_player()
        agent = players[player]
        move = agent.choose(game)
        print(f"{symbol(player)} ({agent.name}) plays {move}")
        game = game.apply(move)
        print(game)
        print()
    winner = game.winner()
    if winner is None:
        print("Result: draw")
    else:
        print(f"Result: {symbol(winner)} wins")
    return winner


def main():
    parser = argparse.ArgumentParser(description="Play a board game in the terminal.")
    parser.add_argument("--board", choices=BOARDS, default="ttt",
                        help="which board to play (default: ttt)")
    parser.add_argument("--x", choices=AGENTS, default="human",
                        help="who plays X / player 1 (default: human)")
    parser.add_argument("--o", choices=AGENTS, default="random",
                        help="who plays O / player 2 (default: random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for random agents, for reproducible games")
    args = parser.parse_args()

    game = BOARDS[args.board]()
    x_agent = AGENTS[args.x](args.seed)
    o_agent = AGENTS[args.o](args.seed)
    play(game, x_agent, o_agent)


if __name__ == "__main__":
    main()
