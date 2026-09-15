"""Step 11: the experiment harness - train while logging the learning curve to CSV.

This is where the headline result comes from. It trains a Q-learning agent and, at regular
checkpoints, freezes it and plays evaluation games against minimax at one or more depths -
writing one row per (checkpoint, depth) to a CSV file:

    games_trained, minimax_depth, wins, draws, losses

The guiding principle: run the EXPENSIVE thing once (training + evaluation) and log the raw
numbers; the CHEAP thing (drawing graphs, Step 12) reads that CSV and can be redone endlessly
without re-running hours of training. Rows are flushed as they're produced, so a long run's
data survives even if you stop it early.

Examples (run from the repo root):

    python experiment.py --board ttt --games 60000 --depths 9
    python experiment.py --board 5x4 --games 300000 --depths 2,4,6 --fold
    python experiment.py --board 5x5 --games 2000000 --depths 2,4 --fold --snapshots
"""

import argparse
import csv
import os
import time

from games import TicTacToe, Connect4
from agents import MinimaxAgent, QLearningAgent, self_play_train, evaluate, save_table

BOARDS = {
    "ttt": lambda: TicTacToe(),
    "5x4": lambda: Connect4(5, 4),
    "5x5": lambda: Connect4(5, 5),
}

FIELDS = ["games_trained", "minimax_depth", "wins", "draws", "losses"]


def run_experiment(make_game, total_games, checkpoint, depths, eval_games=100,
                   fold=False, seed=0, alpha=0.1, gamma=1.0, epsilon=0.1,
                   draw_reward=0.0, opponent_seed=777, csv_path=None,
                   snapshot_dir=None, progress=False):
    """Train, evaluating vs minimax at each checkpoint, logging one row per (checkpoint, depth).

    Returns (agent, rows). If csv_path is given, rows are also written there as they occur.
    """
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon,
                           draw_reward=draw_reward, seed=seed, fold=fold)
    # One fixed minimax opponent per depth (randomized tie-break -> varied, still-perfect lines).
    opponents = [(d, MinimaxAgent(depth=d, seed=opponent_seed)) for d in depths]

    handle = writer = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        handle = open(csv_path, "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

    rows = []
    trained = 0
    while trained < total_games:
        n = min(checkpoint, total_games - trained)
        self_play_train(agent, make_game, n)
        trained += n

        for depth, opponent in opponents:
            w, d, l = evaluate(agent, opponent, make_game, games=eval_games)
            row = {"games_trained": trained, "minimax_depth": depth,
                   "wins": w, "draws": d, "losses": l}
            rows.append(row)
            if writer:
                writer.writerow(row)
                handle.flush()          # survive an early stop
            if progress:
                print(f"  {trained:>9,} games | depth {depth}: {w:>3}W {d:>3}D {l:>3}L")

        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            snap = os.path.join(snapshot_dir, f"snap_{trained}.pkl.gz")
            save_table(snap, agent.q, meta={"fold": fold, "games": trained})

    if handle:
        handle.close()
    return agent, rows


def main():
    parser = argparse.ArgumentParser(description="Log a Q-learning learning curve vs minimax.")
    parser.add_argument("--board", choices=BOARDS, default="ttt")
    parser.add_argument("--games", type=int, default=60000, help="total self-play games")
    parser.add_argument("--checkpoint", type=int, default=None,
                        help="games between log points (default: games/10)")
    parser.add_argument("--depths", default="1,3,5",
                        help="comma-separated minimax depths to test against, e.g. 2,4,6")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="evaluation games per checkpoint per depth")
    parser.add_argument("--fold", action="store_true", help="use symmetry folding")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--draw-reward", type=float, default=0.0)
    parser.add_argument("--snapshots", action="store_true",
                        help="also save a table snapshot at each checkpoint (to snapshots/<board>/)")
    parser.add_argument("--out", default=None, help="CSV path (default: logs/<board>.csv)")
    args = parser.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    checkpoint = args.checkpoint or max(1, args.games // 10)
    out = args.out or f"logs/{args.board}.csv"
    snapshot_dir = f"snapshots/{args.board}" if args.snapshots else None

    print(f"Experiment: {args.board}, {args.games:,} games, checkpoint every {checkpoint:,}, "
          f"depths {depths}, {args.eval_games} eval games each"
          f"{', folded' if args.fold else ''}")
    start = time.time()
    run_experiment(BOARDS[args.board], args.games, checkpoint, depths,
                   eval_games=args.eval_games, fold=args.fold, seed=args.seed,
                   alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                   draw_reward=args.draw_reward, csv_path=out,
                   snapshot_dir=snapshot_dir, progress=True)
    print(f"\nDone in {time.time() - start:.1f}s. Logged -> {out}")
    print("Draw the graphs from it with the Step 12 plotting script.")


if __name__ == "__main__":
    main()
