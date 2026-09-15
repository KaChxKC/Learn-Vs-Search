"""Step 11: the experiment harness logs the right data to CSV.

We don't check the agent's strength here (that's covered elsewhere) - we check the
*logging* is correct: one row per checkpoint per depth, the right columns, and W/D/L that
sum to the number of evaluation games.
"""

import csv

from games.tictactoe import TicTacToe
from experiment import run_experiment, FIELDS


def ttt():
    return TicTacToe()


def test_experiment_writes_one_row_per_checkpoint_with_the_right_columns(tmp_path):
    path = str(tmp_path / "log.csv")
    _, rows = run_experiment(ttt, total_games=2000, checkpoint=1000, depths=[1],
                             eval_games=20, csv_path=path)

    assert len(rows) == 2                       # 2 checkpoints x 1 depth

    with open(path, newline="") as f:
        logged = list(csv.DictReader(f))
    assert len(logged) == 2
    assert list(logged[0].keys()) == FIELDS
    assert logged[0]["games_trained"] == "1000"
    assert logged[1]["games_trained"] == "2000"
    # every checkpoint's W/D/L sums to the number of eval games
    for r in logged:
        assert int(r["wins"]) + int(r["draws"]) + int(r["losses"]) == 20


def test_multiple_depths_produce_a_row_each(tmp_path):
    _, rows = run_experiment(ttt, total_games=1000, checkpoint=500, depths=[1, 2],
                             eval_games=10, csv_path=str(tmp_path / "log.csv"))
    assert len(rows) == 4                        # 2 checkpoints x 2 depths
    assert {r["minimax_depth"] for r in rows} == {1, 2}
