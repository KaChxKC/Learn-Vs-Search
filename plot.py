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
import numpy as np

_INT_FIELDS = ("games_trained", "minimax_depth", "wins", "draws", "losses")


def read_rows(csv_path):
    """Read an experiment CSV into a list of dicts, every column parsed as an int."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: int(v) for k, v in r.items()})
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


def plot_per_seat(rows, out_path):
    """Grouped bars: win rate moving first vs moving second, per minimax depth.

    Uses the final checkpoint of each depth. Needs the per-seat columns from the harness.
    """
    if rows and "first_wins" not in rows[0]:
        raise ValueError("this CSV has no per-seat columns - re-run experiment.py to log them")

    by_depth = defaultdict(list)
    for r in rows:
        by_depth[r["minimax_depth"]].append(r)
    depths = sorted(by_depth)

    first_wr, second_wr = [], []
    for d in depths:
        last = max(by_depth[d], key=lambda r: r["games_trained"])
        ft = max(1, last["first_wins"] + last["first_draws"] + last["first_losses"])
        st = max(1, last["second_wins"] + last["second_draws"] + last["second_losses"])
        first_wr.append(100 * last["first_wins"] / ft)
        second_wr.append(100 * last["second_wins"] / st)

    x = np.arange(len(depths))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, first_wr, w, label="agent moving first", color="tab:blue")
    ax.bar(x + w / 2, second_wr, w, label="agent moving second", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-seat win rate: first vs second mover (final agent)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    _save(fig, out_path)
    return out_path


def _by_depth(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["minimax_depth"]].append(r)
    for d in by:
        by[d].sort(key=lambda r: r["games_trained"])
    return by


def _pct(n, total):
    return 100 * n / max(1, total)


def plot_net_score(rows, out_path, title="Strength over training: net score vs minimax"):
    """Net score (win% - loss%) per minimax depth. 0 = even, above 0 = the learner is ahead."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for depth, pts in sorted(_by_depth(rows).items()):
        xs = [p["games_trained"] for p in pts]
        ys = [_pct(p["wins"] - p["losses"], p["wins"] + p["draws"] + p["losses"]) for p in pts]
        ax.plot(xs, ys, marker="o", label=f"depth {depth}")
    ax.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax.set_xlabel("self-play games trained")
    ax.set_ylabel("net score  (win% - loss%)")
    ax.set_ylim(-100, 100)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="minimax depth")
    _save(fig, out_path)
    return out_path


def plot_outcomes(rows, out_path, title="Outcome mix over training"):
    """Stacked win/draw/loss share over training, one panel per minimax depth."""
    by = _by_depth(rows)
    depths = sorted(by)
    fig, axes = plt.subplots(1, len(depths), figsize=(4.6 * len(depths), 4.6),
                             squeeze=False, sharey=True)
    for ax, depth in zip(axes[0], depths):
        pts = by[depth]
        xs = [p["games_trained"] for p in pts]
        tot = [p["wins"] + p["draws"] + p["losses"] for p in pts]
        ax.stackplot(xs,
                     [_pct(p["wins"], t) for p, t in zip(pts, tot)],
                     [_pct(p["draws"], t) for p, t in zip(pts, tot)],
                     [_pct(p["losses"], t) for p, t in zip(pts, tot)],
                     labels=["win", "draw", "loss"],
                     colors=["tab:green", "tab:gray", "tab:red"], alpha=0.85)
        ax.set_title(f"vs minimax depth {depth}")
        ax.set_xlabel("games trained")
        ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("share of games (%)")
    axes[0][-1].legend(loc="upper right")
    fig.suptitle(title)
    _save(fig, out_path)
    return out_path


def plot_depth_scaling(rows, out_path):
    """Final agent's win/draw/loss share against each minimax depth (needs 2+ depths)."""
    by = _by_depth(rows)
    depths = sorted(by)
    if len(depths) < 2:
        raise ValueError("depth scaling needs at least two minimax depths in the CSV")
    last = [by[d][-1] for d in depths]
    tot = [p["wins"] + p["draws"] + p["losses"] for p in last]
    win = [_pct(p["wins"], t) for p, t in zip(last, tot)]
    draw = [_pct(p["draws"], t) for p, t in zip(last, tot)]
    loss = [_pct(p["losses"], t) for p, t in zip(last, tot)]
    x = np.arange(len(depths))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, win, color="tab:green", label="win")
    ax.bar(x, draw, bottom=win, color="tab:gray", label="draw")
    ax.bar(x, loss, bottom=[a + b for a, b in zip(win, draw)], color="tab:red", label="loss")
    ax.set_xticks(x)
    ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylabel("share of games (%)")
    ax.set_ylim(0, 100)
    games = last[0]["games_trained"]
    ax.set_title(f"Trained agent vs deeper search ({games:,} training games)")
    ax.legend(loc="upper right")
    _save(fig, out_path)
    return out_path


def plot_coverage(rows, out_path, title="Coverage: how much of the game the learner has seen"):
    """Left: table size vs games. Right: strength vs table size (does more coverage = better?)."""
    if not rows or "states" not in rows[0]:
        raise ValueError("this CSV has no 'states' column - re-run experiment.py to log it")
    by = _by_depth(rows)
    depths = sorted(by)
    base = by[depths[0]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    a1.plot([p["games_trained"] for p in base], [p["states"] for p in base], marker="o", color="tab:blue")
    a1.set_xlabel("self-play games trained")
    a1.set_ylabel("states in the Q-table")
    a1.set_title("Table growth")
    a1.grid(True, alpha=0.3)
    for depth in depths:
        pts = by[depth]
        a2.plot([p["states"] for p in pts],
                [_pct(p["wins"] - p["losses"], p["wins"] + p["draws"] + p["losses"]) for p in pts],
                marker="o", label=f"depth {depth}")
    a2.axhline(0, color="black", lw=0.8, alpha=0.6)
    a2.set_xlabel("states in the Q-table")
    a2.set_ylabel("net score  (win% - loss%)")
    a2.set_title("Strength vs coverage")
    a2.grid(True, alpha=0.3)
    a2.legend(title="minimax depth")
    fig.suptitle(title)
    _save(fig, out_path)
    return out_path


def plot_seat_curves(rows, out_path):
    """Win rate over training when the learner moves first (solid) vs second (dashed)."""
    if not rows or "first_wins" not in rows[0]:
        raise ValueError("this CSV has no per-seat columns - re-run experiment.py to log them")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (depth, pts) in enumerate(sorted(_by_depth(rows).items())):
        color = f"C{i}"
        xs = [p["games_trained"] for p in pts]
        ax.plot(xs, [_pct(p["first_wins"], p["first_wins"] + p["first_draws"] + p["first_losses"]) for p in pts],
                color=color, marker="o", label=f"depth {depth}, moving first")
        ax.plot(xs, [_pct(p["second_wins"], p["second_wins"] + p["second_draws"] + p["second_losses"]) for p in pts],
                color=color, marker="s", linestyle="--", label=f"depth {depth}, moving second")
    ax.set_xlabel("self-play games trained")
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("First-move advantage over training")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_path)
    return out_path


def plot_board_comparison(csv_paths, out_path):
    """Final net score of the trained agent on each board, per minimax depth."""
    labels, scores, colors = [], [], []
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for i, path in enumerate(csv_paths):
        name = os.path.splitext(os.path.basename(path))[0]
        for depth, pts in sorted(_by_depth(read_rows(path)).items()):
            last = pts[-1]
            total = last["wins"] + last["draws"] + last["losses"]
            labels.append(f"{name}\ndepth {depth}")
            scores.append(_pct(last["wins"] - last["losses"], total))
            colors.append(palette[i % len(palette)])
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 5))
    ax.bar(range(len(labels)), scores, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("net score  (win% - loss%)")
    ax.set_ylim(-100, 100)
    ax.set_title("How the trained agent fares as the board gets bigger")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path)
    return out_path


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all(csv_path, learning_curve_out=None):
    """Draw every figure that this experiment CSV has the data for. Returns the paths drawn."""
    rows = read_rows(csv_path)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    drawn = []

    out = learning_curve_out or f"figures/{name}_learning_curve.png"
    plot_learning_curve(rows, out, title=f"Learning curve ({name}): Q-agent vs minimax")
    print(f"Saved learning curve -> {out}")
    drawn.append(out)

    extras = [
        ("net_score", plot_net_score, True),
        ("outcomes", plot_outcomes, True),
        ("depth_scaling", plot_depth_scaling, len({r["minimax_depth"] for r in rows}) > 1),
        ("per_seat", plot_per_seat, bool(rows) and "first_wins" in rows[0]),
        ("seat_curves", plot_seat_curves, bool(rows) and "first_wins" in rows[0]),
        ("coverage", plot_coverage, bool(rows) and "states" in rows[0]),
    ]
    for suffix, fn, applicable in extras:
        if not applicable:
            print(f"Skipped {suffix} (this CSV doesn't have the data for it)")
            continue
        path = f"figures/{name}_{suffix}.png"
        fn(rows, path)
        print(f"Saved {suffix} -> {path}")
        drawn.append(path)
    return drawn


def main():
    parser = argparse.ArgumentParser(description="Draw figures from experiment CSVs.")
    parser.add_argument("--csv", help="experiment CSV to draw a learning curve from")
    parser.add_argument("--alpha-beta", dest="alpha_beta",
                        choices=["ttt", "5x4", "5x5"],
                        help="draw the alpha-beta efficiency graph for this board")
    parser.add_argument("--compare", nargs="+", metavar="CSV",
                        help="compare final results across several experiment CSVs (boards)")
    parser.add_argument("--out", default=None, help="output PNG path")
    args = parser.parse_args()

    if not args.csv and not args.alpha_beta and not args.compare:
        parser.error("give --csv <file>, --alpha-beta <board>, and/or --compare <csv> ...")

    if args.compare:
        out = args.out or "figures/board_comparison.png"
        plot_board_comparison(args.compare, out)
        print(f"Saved board comparison -> {out}")

    if args.csv:
        plot_all(args.csv, learning_curve_out=args.out)

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
