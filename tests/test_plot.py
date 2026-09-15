"""Step 12: the plotting script turns CSV rows into image files.

We check the plumbing - that reading a CSV and drawing produces a non-empty PNG - not the
pixels. Uses a tiny synthetic CSV so it needs no real experiment run.
"""

import csv
import os

from games.tictactoe import TicTacToe
from plot import read_rows, plot_learning_curve, plot_alpha_beta, _INT_FIELDS


def _write_csv(path):
    rows = [
        {"games_trained": 1000, "minimax_depth": 3, "wins": 20, "draws": 30, "losses": 50},
        {"games_trained": 2000, "minimax_depth": 3, "wins": 40, "draws": 40, "losses": 20},
        {"games_trained": 1000, "minimax_depth": 5, "wins": 10, "draws": 30, "losses": 60},
        {"games_trained": 2000, "minimax_depth": 5, "wins": 25, "draws": 45, "losses": 30},
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_INT_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_read_rows_parses_integers(tmp_path):
    path = str(tmp_path / "log.csv")
    _write_csv(path)
    rows = read_rows(path)
    assert len(rows) == 4
    assert rows[0]["wins"] == 20 and isinstance(rows[0]["games_trained"], int)


def test_learning_curve_writes_a_png(tmp_path):
    csv_path = str(tmp_path / "log.csv")
    _write_csv(csv_path)
    out = str(tmp_path / "curve.png")
    plot_learning_curve(read_rows(csv_path), out)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_alpha_beta_writes_a_png(tmp_path):
    out = str(tmp_path / "ab.png")
    plot_alpha_beta(lambda: TicTacToe(), out, max_depth=4)
    assert os.path.exists(out) and os.path.getsize(out) > 0
