# Learn-Vs-Search

A comparative study of two ways to play Connect-4: an agent that **learns** the game
purely from experience, and an agent that **searches** for the best move by reasoning
ahead. Both are built from scratch in plain Python and pitted against each other on
scaled Connect-4 boards.

- **Learn** — tabular **Q-learning**, trained only through self-play. It starts knowing
  nothing and builds a table of how good each position is by playing millions of games
  against itself.
- **Search** — depth-limited **minimax with alpha-beta pruning**, the classic adversarial
  search algorithm. It never "learns"; it looks a fixed number of moves ahead every time.

## The research question

> **How many self-play games does the Q-learning agent need before it can match
> depth-limited minimax?**

This framing makes the project robust: minimax always works, so even if the learning
agent never catches up, *that* is a complete, measurable result — "the learner needed
more than N games to reach depth-d search, and here is the evidence." An honest mixed or
negative finding scores as well as a win. There is no single point of failure.

## The board ladder

The same game engine runs three boards. Each one proves something the next depends on, so
any bug stays local to the step it appears in rather than hiding several phases back.

| Board | Size | Role |
|-------|------|------|
| **Tic-tac-toe** | 3×3 | Proof-of-life. A *solved* game with known-correct answers: full-depth minimax must never lose, and a trained learner must stop losing. The cheapest place to catch bugs. |
| **Connect-4 (small)** | 5×4 | Validates the real Connect-4 mechanics — gravity, four-in-a-row, diagonal wins, full columns — and trains fast enough that bugs surface in minutes. |
| **Connect-4 (main)** | 5×5 | The headline experiment board. Big enough for rich tactics, still tractable for a table-based method. |

Four-in-a-row throughout. The larger 6×5 and full 7×6 boards are deliberately **avoided** —
their state space is too large for a tabular method and would force a neural-net approach,
which is out of scope.

## How it fits together

Everything hinges on one small idea: **agents never know which board they are playing.**
Every board satisfies a single `Game` interface, so the search and learning code is
written *once* and runs on all three boards unchanged.

```
Game  (what every board provides)                Agent  (what every player provides)
  legal_moves()   -> list of integer moves          choose(game) -> one legal move
  apply(move)     -> a NEW game state
  winner()        -> P1 / P2 / None               Implementations:
  current_player()                                   RandomAgent
  state_key()     -> hashable Q-table key            MinimaxAgent(depth)
  is_terminal() / is_draw()                          QLearningAgent(qtable)
                                                     HumanAgent
```

Two design choices keep the code understandable and bug-resistant:

- **A move is always a plain integer** — a cell index for tic-tac-toe, a column for
  Connect-4 — so no agent ever special-cases a board.
- **Boards are immutable**: `apply(move)` returns a *new* game instead of changing the
  current one. This removes the whole category of "forgot to undo that move" bugs from the
  search and learning loops.

## Project layout

```
games/       board engines — the Game interface (base.py) and each board
agents/       the players: random, minimax, q-learning, human
tests/        hand-checked correctness tests (run with pytest)
train.py      run self-play training; produces a Q-table file
play.py       play in the terminal (a debugging tool, not a deliverable)
ui.py         a thin Pygame viewer over the finished agents (built last)
```

The Pygame window is only a **viewer** — it loads a trained table and calls each agent's
`choose(game)`. No game rules or AI live in it, which is why it is built last and is the
safe thing to cut if time runs short. Planned modes: human-vs-learner, human-vs-search,
and a learner-vs-search watch mode, with board, first-move, and search-depth selectors,
plus a restart button.

## Results and graphs

The analysis lives in five figures, drawn by a plotting script from logged data (so
experiments run once and plots are redrawn as often as needed):

1. **Learning curve (headline)** — win/draw/loss rate vs. minimax as training progresses;
   answers the research question.
2. **Alpha-beta efficiency** — nodes searched by plain minimax vs. alpha-beta at each
   depth; proof the pruning is correct.
3. **Per-seat win rates** — first-mover vs. second-mover, quantifying (and controlling
   for) the first-move advantage on a small board.
4. **Hyperparameter sweeps** — how learning rate, exploration decay, and draw reward
   affect convergence.
5. **Depth-scaling (optional)** — how the trained learner holds up as the searcher looks
   deeper.

Fairness matters on small boards, where going first is a real advantage: evaluation
alternates the starting side exactly 50/50 so the numbers reflect skill, not seating.

## Setup

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate         # Windows (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
pytest                          # run the correctness tests
```

## Trained tables

Trained Q-tables are **not** stored in git history — they are large binary files that
would bloat the repository. Instead:

- code, logs (CSV/JSON), and generated plots live in git;
- large trained tables are distributed as **GitHub Release** assets;
- tables are gzip-compressed, and a small demo table may be committed so the UI runs
  straight after cloning.

## Roadmap

- [x] **Step 0** — repository skeleton
- [x] **Step 1** — the `Game` interface + tic-tac-toe
- [x] **Step 2** — the Connect-4 board (5×4 and 5×5)
- [x] **Step 3** — hand-checked win-detection tests for Connect-4
- [x] **Step 4** — random agent + terminal play
- [x] **Step 5** — minimax
- [ ] **Step 6** — alpha-beta pruning + node counting
- [ ] **Step 7** — Q-learning core + self-play loop
- [ ] **Step 8** — validate the learner on tic-tac-toe
- [ ] **Step 9** — scale to 5×4, beat a random player
- [ ] **Step 10** — scale to 5×5 + symmetry folding
- [ ] **Step 11** — experiment harness + CSV/JSON logging
- [ ] **Step 12** — plotting script for all figures
- [ ] **Step 13** — hyperparameter sweeps
- [ ] **Step 14** — Pygame UI
