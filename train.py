"""Train a Q-learning agent by self-play and save its table.

Examples (run from the repo root):

    python train.py --board ttt --games 50000
    python train.py --board 5x4 --games 200000 --epsilon 0.2
    python train.py --board 5x5 --games 1000000 --out qtable_5x5.pkl.gz

While it trains, it periodically plays the (greedy) agent against a random opponent and
prints win/draw/loss - a quick sanity signal that learning is happening. This console
readout is NOT the formal experiment logging (that's the Step 11 harness); it's just
so you can watch progress. When done it saves the learned table (gzip-compressed pickle).
"""

import argparse
import gzip
import pickle
import time

from games import TicTacToe, Connect4
from agents import RandomAgent, QLearningAgent, self_play_train, evaluate

BOARDS = {
    "ttt": lambda: TicTacToe(),
    "5x4": lambda: Connect4(5, 4),
    "5x5": lambda: Connect4(5, 5),
}


def main():
    parser = argparse.ArgumentParser(description="Train a Q-learning agent by self-play.")
    parser.add_argument("--board", choices=BOARDS, default="ttt")
    parser.add_argument("--games", type=int, default=50000, help="self-play games to train on")
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration rate")
    parser.add_argument("--draw-reward", type=float, default=0.0, help="value of a draw")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoints", type=int, default=10,
                        help="how many progress readouts to print during training")
    parser.add_argument("--out", default=None, help="output file (default: qtable_<board>.pkl.gz)")
    args = parser.parse_args()

    make = BOARDS[args.board]
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                           draw_reward=args.draw_reward, seed=args.seed)
    ruler = RandomAgent(seed=12345)

    chunk = max(1, args.games // args.checkpoints)
    trained = 0
    print(f"Training Q-learning on {args.board} for {args.games:,} games...")
    print(f"{'games':>10} | {'states':>8} | {'vs random (W/D/L)':>18} | {'win%':>5}")
    start = time.time()
    while trained < args.games:
        n = min(chunk, args.games - trained)
        self_play_train(agent, make, n)
        trained += n
        w, d, l = evaluate(agent, ruler, make, games=200)
        print(f"{trained:>10,} | {len(agent.q):>8,} | {w:>5}/{d:>4}/{l:>5}      | {100*w/(w+d+l):>4.0f}%")

    elapsed = time.time() - start
    out = args.out or f"qtable_{args.board}.pkl.gz"
    with gzip.open(out, "wb") as f:
        pickle.dump(agent.q, f)
    print(f"\nDone in {elapsed:.1f}s. Learned {len(agent.q):,} states.")
    print(f"Saved table -> {out}")
    print(f"Play it:  python play.py --board {args.board} --x human --o qlearn --load {out}")


if __name__ == "__main__":
    main()
