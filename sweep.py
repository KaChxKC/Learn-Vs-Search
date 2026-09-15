"""Step 13: hyperparameter sweeps - how a setting affects the learner's convergence.

This is framed as investigation, not a config you get "right" once. It runs the Step 11
experiment several times, varying ONE hyperparameter (learning rate, exploration, or draw
reward) across a set of values, and overlays the resulting curves so you can see which
value converges faster or to a stronger level.

The y-axis is a single "net score" = win% - loss% against a fixed minimax depth. That one
number works on every board: on tic-tac-toe vs perfect play it rises from negative (losing)
toward 0 (drawing every game); on bigger boards it rises as the learner starts winning.

    python sweep.py --board ttt --param alpha   --values 0.05,0.1,0.3 --games 60000 --depth 9
    python sweep.py --board 5x4 --param epsilon --values 0.1,0.2,0.4 --games 300000 --depth 4 --fold
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import run_experiment, BOARDS

PARAMS = ("alpha", "epsilon", "draw_reward")


def run_sweep(make_game, param, values, total_games, checkpoint, depth,
              eval_games=100, fold=False, seed=0, progress=False):
    """Run one experiment per value of `param`. Returns {value: rows}."""
    assert param in PARAMS, f"param must be one of {PARAMS}"
    results = {}
    for v in values:
        kwargs = {"alpha": 0.1, "epsilon": 0.1, "draw_reward": 0.0}
        kwargs[param] = v
        if progress:
            print(f"[{param} = {v}]")
        _, rows = run_experiment(make_game, total_games, checkpoint, [depth],
                                 eval_games=eval_games, fold=fold, seed=seed,
                                 progress=progress, **kwargs)
        results[v] = rows
    return results


def net_score(row):
    """win% - loss%: a single 'strength' number that works on every board."""
    total = max(1, row["wins"] + row["draws"] + row["losses"])
    return 100 * (row["wins"] - row["losses"]) / total


def write_sweep_csv(results, param, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([param, "games_trained", "wins", "draws", "losses"])
        for v in sorted(results):
            for r in sorted(results[v], key=lambda r: r["games_trained"]):
                w.writerow([v, r["games_trained"], r["wins"], r["draws"], r["losses"]])


def plot_sweep(results, param, out_path, depth=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(results):
        rows = sorted(results[v], key=lambda r: r["games_trained"])
        xs = [r["games_trained"] for r in rows]
        ys = [net_score(r) for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{param} = {v}")

    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("self-play games trained")
    ax.set_ylabel("net score vs minimax  (win% - loss%)")
    title = f"Effect of {param} on convergence"
    if depth is not None:
        title += f"  (vs minimax depth {depth})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title=param)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Sweep a hyperparameter and plot convergence.")
    parser.add_argument("--board", choices=BOARDS, default="ttt")
    parser.add_argument("--param", choices=PARAMS, required=True)
    parser.add_argument("--values", required=True, help="comma-separated values, e.g. 0.05,0.1,0.3")
    parser.add_argument("--games", type=int, default=60000)
    parser.add_argument("--checkpoint", type=int, default=None, help="default: games/10")
    parser.add_argument("--depth", type=int, default=None, help="minimax depth (default: 9 ttt, else 4)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--fold", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="CSV path (default: logs/sweep_<board>_<param>.csv)")
    parser.add_argument("--fig", default=None, help="PNG path (default: figures/sweep_<board>_<param>.png)")
    args = parser.parse_args()

    values = [float(x) for x in args.values.split(",") if x.strip()]
    checkpoint = args.checkpoint or max(1, args.games // 10)
    depth = args.depth if args.depth is not None else (9 if args.board == "ttt" else 4)
    out = args.out or f"logs/sweep_{args.board}_{args.param}.csv"
    fig = args.fig or f"figures/sweep_{args.board}_{args.param}.png"

    print(f"Sweeping {args.param} over {values} on {args.board} "
          f"({args.games:,} games, vs minimax depth {depth})")
    results = run_sweep(BOARDS[args.board], args.param, values, args.games, checkpoint,
                        depth, eval_games=args.eval_games, fold=args.fold,
                        seed=args.seed, progress=True)
    write_sweep_csv(results, args.param, out)
    plot_sweep(results, args.param, fig, depth=depth)
    print(f"\nSaved data -> {out}")
    print(f"Saved figure -> {fig}")


if __name__ == "__main__":
    main()
