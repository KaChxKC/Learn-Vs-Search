"""Step 13: the hyperparameter sweep runs one experiment per value and plots them.

We check the plumbing on a tiny fast sweep: one result series per value, a CSV with the
right rows, and a non-empty figure. Uses synthetic-scale runs so it stays quick.
"""

import csv
import os

from games.tictactoe import TicTacToe
from sweep import run_sweep, write_sweep_csv, plot_sweep, net_score


def ttt():
    return TicTacToe()


def test_run_sweep_produces_one_series_per_value():
    results = run_sweep(ttt, "alpha", [0.1, 0.5], total_games=1000, checkpoint=500,
                        depth=1, eval_games=10)
    assert set(results) == {0.1, 0.5}
    for rows in results.values():
        assert len(rows) == 2                    # 2 checkpoints, single depth


def test_net_score_is_win_minus_loss_percent():
    assert net_score({"wins": 60, "draws": 20, "losses": 20}) == 40.0
    assert net_score({"wins": 0, "draws": 80, "losses": 20}) == -20.0


def test_sweep_writes_csv_and_figure(tmp_path):
    results = run_sweep(ttt, "epsilon", [0.1, 0.3], total_games=1000, checkpoint=500,
                        depth=1, eval_games=10)

    csv_path = str(tmp_path / "sweep.csv")
    write_sweep_csv(results, "epsilon", csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4                          # 2 values x 2 checkpoints
    assert "epsilon" in rows[0]

    fig_path = str(tmp_path / "sweep.png")
    plot_sweep(results, "epsilon", fig_path)
    assert os.path.exists(fig_path) and os.path.getsize(fig_path) > 0
