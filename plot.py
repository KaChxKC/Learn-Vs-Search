"""Step 12: draw the figures from the experiment CSVs.

Plotting is the CHEAP half of the "log once, plot often" split. This reads the raw numbers
experiment.py logged (games_trained, minimax_depth, wins, draws, losses) and draws graphs -
so restyling or relabelling never means re-running training.

    python plot.py --csv logs/ttt.csv                 # learning curve from a CSV
    python plot.py --csv logs/5x4.csv --out figures/5x4.png
    python plot.py --alpha-beta 5x4                    # alpha-beta efficiency (no CSV needed)
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")           # render to files, no display window needed
import matplotlib.pyplot as plt

_INT_FIELDS = ("games_trained", "minimax_depth", "wins", "draws", "losses")


def read_rows(csv_path):
    """Read an experiment CSV into a list of dicts with integer values."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: int(r[k]) for k in _INT_FIELDS})
    return rows


def _rates(pts):
    """(xs, win%, draw%, loss%) for checkpoints sorted by games trained."""
    pts = sorted(pts, key=lambda r: r["games_trained"])
    xs = [p["games_trained"] for p in pts]
    total = [max(1, p["wins"] + p["draws"] + p["losses"]) for p in pts]
    win = [100 * p["wins"] / t for p, t in zip(pts, total)]
    draw = [100 * p["draws"] / t for p, t in zip(pts, total)]
    loss = [100 * p["losses"] / t for p, t in zip(pts, total)]
    return xs, win, draw, loss


def plot_learning_curve(rows, out_path, title="Learning curve: Q-agent vs minimax"):
    """The headline graph.

    With ONE minimax depth: plot win/draw/loss rate (the full story, e.g. losses falling to
    zero against perfect play). With SEVERAL depths: plot win rate per depth, to compare how
    the learner fares against deeper search.
    """
    by_depth = defaultdict(list)
    for r in rows:
        by_depth[r["minimax_depth"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(by_depth) == 1:
        (depth, pts), = by_depth.items()
        xs, win, draw, loss = _rates(pts)
        ax.plot(xs, win, marker="o", color="tab:green", label="win %")
        ax.plot(xs, draw, marker="o", color="tab:gray", label="draw %")
        ax.plot(xs, loss, marker="o", color="tab:red", label="loss %")
        ax.set_ylabel("outcome rate vs minimax (%)")
        ax.legend(title=f"vs minimax depth {depth}")
    else:
        for depth in sorted(by_depth):
            xs, win, _, _ = _rates(by_depth[depth])
            ax.plot(xs, win, marker="o", label=f"depth {depth}")
        ax.set_ylabel("win rate vs minimax (%)")
        ax.legend(title="minimax depth")

    ax.set_xlabel("self-play games trained")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    _save(fig, out_path)
    return out_path


def plot_alpha_beta(make_game, out_path, max_depth=7):
    """The 'almost free' graph: nodes searched by plain minimax vs alpha-beta, per depth."""
    from agents import MinimaxAgent

    depths = list(range(1, max_depth + 1))
    plain_nodes, ab_nodes = [], []
    for d in depths:
        plain = MinimaxAgent(depth=d, prune=False)
        ab = MinimaxAgent(depth=d, prune=True)
        plain.choose(make_game())
        ab.choose(make_game())
        plain_nodes.append(plain.nodes)
        ab_nodes.append(ab.nodes)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(depths, plain_nodes, marker="o", label="plain minimax")
    ax.plot(depths, ab_nodes, marker="o", label="alpha-beta")
    ax.set_xlabel("search depth")
    ax.set_ylabel("positions examined")
    ax.set_yscale("log")            # the gap spans orders of magnitude
    ax.set_title("Alpha-beta efficiency: positions searched vs depth")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _save(fig, out_path)
    return out_path


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw figures from experiment CSVs.")
    parser.add_argument("--csv", help="experiment CSV to draw a learning curve from")
    parser.add_argument("--alpha-beta", dest="alpha_beta",
                        choices=["ttt", "5x4", "5x5"],
                        help="draw the alpha-beta efficiency graph for this board")
    parser.add_argument("--out", default=None, help="output PNG path")
    args = parser.parse_args()

    if not args.csv and not args.alpha_beta:
        parser.error("give --csv <file> and/or --alpha-beta <board>")

    if args.csv:
        rows = read_rows(args.csv)
        name = os.path.splitext(os.path.basename(args.csv))[0]
        out = args.out or f"figures/{name}_learning_curve.png"
        plot_learning_curve(rows, out, title=f"Learning curve ({name}): Q-agent vs minimax")
        print(f"Saved learning curve -> {out}")

    if args.alpha_beta:
        from games import TicTacToe, Connect4
        makers = {"ttt": lambda: TicTacToe(),
                  "5x4": lambda: Connect4(5, 4),
                  "5x5": lambda: Connect4(5, 5)}
        out = args.out or f"figures/alpha_beta_{args.alpha_beta}.png"
        plot_alpha_beta(makers[args.alpha_beta], out)
        print(f"Saved alpha-beta graph -> {out}")


if __name__ == "__main__":
    main()
