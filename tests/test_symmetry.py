"""Step 10: symmetry folding + table save/load with metadata.

Connect-4 has a left-right mirror symmetry, so a position and its mirror can share one
table entry. Folding roughly halves the states a learner must cover and doubles how much
each game teaches. These tests confirm the folding is correct (mirror boards map to the
same key) and that it actually shrinks the table.
"""

import gzip
import pickle

from games.connect4 import Connect4
from agents.qlearn import QLearningAgent, self_play_train, save_table, load_table


def c5x4():
    return Connect4(5, 4)


def test_mirror_boards_share_one_canonical_key():
    # Play some moves on the left; play the mirrored moves (col c <-> 3-c) on another board.
    left = c5x4().apply(0).apply(1).apply(0)
    right = c5x4().apply(3).apply(2).apply(3)
    assert left.canonical()[0] == right.canonical()[0]     # same folded key


def test_folding_shrinks_the_state_count():
    plain = QLearningAgent(seed=0, fold=False)
    self_play_train(plain, c5x4, games=4000)
    folded = QLearningAgent(seed=0, fold=True)
    self_play_train(folded, c5x4, games=4000)
    assert len(folded.q) < len(plain.q)                    # fewer states, same games


def test_folded_agent_plays_legal_moves():
    agent = QLearningAgent(seed=0, fold=True)
    self_play_train(agent, c5x4, games=2000)
    game = c5x4()
    assert agent.choose(game) in game.legal_moves()


def test_save_load_roundtrip_preserves_table_and_fold_flag(tmp_path):
    agent = QLearningAgent(seed=0, fold=True)
    self_play_train(agent, c5x4, games=1000)
    path = str(tmp_path / "table.pkl.gz")

    save_table(path, agent.q, meta={"fold": True, "board": "5x4"})
    q, meta = load_table(path)

    assert meta.get("fold") is True
    assert len(q) == len(agent.q)


def test_load_accepts_a_legacy_raw_table(tmp_path):
    # Tables saved before metadata existed are bare dicts; they must still load.
    raw = {("board",): {0: 1.0}}
    path = str(tmp_path / "legacy.pkl.gz")
    with gzip.open(path, "wb") as f:
        pickle.dump(raw, f)

    q, meta = load_table(path)
    assert q == raw
    assert meta == {}
