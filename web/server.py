"""Step 15: a small local web server for a shareable, browser-playable demo.

Like the Pygame UI, this is ONLY a viewer - it has no game rules and no AI of its own. The
browser draws the board and sends clicks; this server reuses the existing Game engine and
Agents to compute moves. Game logic stays in Python; the page is just a display.

Run it:
    pip install flask          # one-time (also in requirements.txt)
    python web/server.py
    # then open http://127.0.0.1:5000 in your browser

Endpoints (all JSON):
    POST /api/new   {board, red, yellow, depth}  -> fresh game state + game_id
    POST /api/move  {game_id, move}              -> apply a human move, return state
    POST /api/step  {game_id}                    -> let the current AI play one move
"""

import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)            # repo root, where qtable_*.pkl.gz and games/ live
sys.path.insert(0, ROOT)               # so `import games` works when run from anywhere

from flask import Flask, request, jsonify, send_from_directory

from games import TicTacToe, Connect4, P1, P2
from agents import RandomAgent, MinimaxAgent, QLearningAgent, load_table
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="")

BOARDS = {
    "ttt": (lambda: TicTacToe(), (3, 3)),
    "5x4": (lambda: Connect4(5, 4), (5, 4)),
    "5x5": (lambda: Connect4(5, 5), (5, 5)),
}

games = {}          # game_id -> {game, board, agents, kinds, note}


def make_agent(kind, board_key, depth):
    if kind == "Human":
        return None, ""
    if kind == "Random":
        return RandomAgent(), ""
    if kind == "Minimax":
        depth = None if board_key == "ttt" else depth
        return MinimaxAgent(depth=depth, seed=1), ""
    if kind == "Q-learn":
        path = os.path.join(ROOT, f"qtable_{board_key}.pkl.gz")
        if os.path.exists(path):
            q, meta = load_table(path)
            return QLearningAgent(epsilon=0.0, q=q, fold=meta.get("fold", False)), ""
        return RandomAgent(), f"No qtable_{board_key}.pkl.gz - train it first; using Random."
    return RandomAgent(), ""


def winning_cells(game, rows, cols, flat):
    """Cells that form the winning line(s), purely so the page can highlight them."""
    winner = game.winner()
    if winner is None:
        return []
    need = getattr(game, "connect", 3)
    cells = set()
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                run = [(r + i * dr, c + i * dc) for i in range(need)]
                if all(0 <= rr < rows and 0 <= cc < cols and flat[rr * cols + cc] == winner
                       for rr, cc in run):
                    cells.update(run)
    return sorted([r, c] for r, c in cells)


def last_cell(rec, rows, cols, grid):
    """(row, col) where the most recent move landed, so the page can animate it."""
    if not rec["history"]:
        return None
    move = rec["history"][-1]
    if rec["board"] == "ttt":
        return [move // cols, move % cols]
    for r in range(rows):                    # gravity: the topmost disc in that column
        if grid[r][move]:
            return [r, move]
    return None


def serialize(rec):
    game = rec["game"]
    rows, cols = BOARDS[rec["board"]][1]
    flat = game.state_key()[0]
    grid = [[flat[r * cols + c] for c in range(cols)] for r in range(rows)]
    current = None if game.is_terminal() else game.current_player()
    kinds = rec["kinds"]
    return {
        "board": rec["board"], "rows": rows, "cols": cols, "grid": grid,
        "gravity": rec["board"] != "ttt",
        "current": current,
        "current_is_human": current is not None and kinds[current] == "Human",
        "legal": game.legal_moves(),
        "terminal": game.is_terminal(),
        "winner": game.winner(),
        "win_cells": winning_cells(game, rows, cols, flat),
        "history": rec["history"],
        "last_cell": last_cell(rec, rows, cols, grid),
        "red_kind": kinds[P1], "yellow_kind": kinds[P2],
        "note": rec.get("note", ""),
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/new")
def api_new():
    data = request.get_json(force=True)
    board = data.get("board", "ttt")
    if board not in BOARDS:
        return jsonify(error="unknown board"), 400
    red, yellow = data.get("red", "Human"), data.get("yellow", "Minimax")
    depth = int(data.get("depth", 3))

    red_agent, note1 = make_agent(red, board, depth)
    yellow_agent, note2 = make_agent(yellow, board, depth)
    game_id = uuid.uuid4().hex
    games[game_id] = {
        "game": BOARDS[board][0](),
        "board": board,
        "agents": {P1: red_agent, P2: yellow_agent},
        "kinds": {P1: red, P2: yellow},
        "note": note1 or note2,
        "history": [],
    }
    state = serialize(games[game_id])
    state["game_id"] = game_id
    return jsonify(state)


@app.post("/api/move")
def api_move():
    rec = games.get(request.get_json(force=True).get("game_id"))
    if rec is None:
        return jsonify(error="unknown game"), 404
    game = rec["game"]
    if game.is_terminal():
        return jsonify(serialize(rec))
    if rec["agents"][game.current_player()] is not None:
        return jsonify(error="not a human's turn"), 400
    move = int(request.get_json(force=True)["move"])
    if move not in game.legal_moves():
        return jsonify(error="illegal move"), 400
    rec["history"].append(move)
    rec["game"] = game.apply(move)
    return jsonify(serialize(rec))


@app.post("/api/step")
def api_step():
    rec = games.get(request.get_json(force=True).get("game_id"))
    if rec is None:
        return jsonify(error="unknown game"), 404
    game = rec["game"]
    if not game.is_terminal():
        agent = rec["agents"][game.current_player()]
        if agent is not None:
            move = agent.choose(game)
            rec["history"].append(move)
            rec["game"] = game.apply(move)
    return jsonify(serialize(rec))


FIG_DIR = os.path.join(ROOT, "figures")


@app.get("/api/figures")
def api_figures():
    """The PNG graphs in figures/ (newest changes show up without restarting the server)."""
    items = []
    if os.path.isdir(FIG_DIR):
        for name in sorted(os.listdir(FIG_DIR)):
            if name.lower().endswith(".png"):
                mtime = int(os.path.getmtime(os.path.join(FIG_DIR, name)))
                items.append({"name": name, "url": f"/figures/{name}?v={mtime}"})
    return jsonify(figures=items)


@app.get("/figures/<path:name>")
def figure_file(name):
    return send_from_directory(FIG_DIR, name)


@app.get("/api/info")
def api_info():
    """Which boards have a trained Q-table on disk (so the page can say so)."""
    return jsonify(tables={
        key: os.path.exists(os.path.join(ROOT, f"qtable_{key}.pkl.gz")) for key in BOARDS
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
