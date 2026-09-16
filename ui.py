"""Step 14: a thin Pygame viewer over the finished agents.

It is ONLY a viewer - no game rules and no AI of its own. It builds a Game and two Agents
(exactly like play.py), draws the board, forwards human clicks as moves, and steps the AI
players. Everything it shows already works without it, which is why it comes last and is the
safe thing to cut if time runs short.

Run it:
    python ui.py

Top-bar controls (click to change; changing any of them starts a fresh game):
    Board:  TTT / 5x4 / 5x5
    X , O:  Human / Random / Minimax / Q-learn   (Q-learn loads qtable_<board>.pkl.gz)
    Depth:  minimax search depth  [-] [+]
    Restart

Click a column (Connect-4) or a cell (tic-tac-toe) to make a human move. When it is an AI's
turn it plays by itself after a short pause, so you can watch algorithm-vs-algorithm games.
"""

import os
from collections import namedtuple

import pygame

from games import TicTacToe, Connect4, P1, P2
from agents import RandomAgent, MinimaxAgent, QLearningAgent, load_table

WIDTH, HEIGHT = 600, 640
PANEL_H = 96
AI_DELAY_MS = 400

BG = (28, 28, 36)
PANEL = (44, 44, 58)
GRID = (70, 90, 120)
EMPTY = (22, 22, 30)
TEXT = (232, 232, 238)
BTN = (60, 70, 96)
BTN_HI = (92, 112, 156)
DISC = {P1: (222, 74, 74), P2: (242, 202, 66)}     # red = player 1, yellow = player 2

BOARDS = {                        # label -> (board key, builder, (rows, cols))
    "TTT": ("ttt", lambda: TicTacToe(), (3, 3)),
    "5x4": ("5x4", lambda: Connect4(5, 4), (5, 4)),
    "5x5": ("5x5", lambda: Connect4(5, 5), (5, 5)),
}
BOARD_ORDER = ["TTT", "5x4", "5x5"]
PLAYER_KINDS = ["Human", "Random", "Minimax", "Q-learn"]

Geom = namedtuple("Geom", "x0 y0 cell rows cols")


def pixel_to_cell(g, px, py):
    """Map a pixel to a (row, col) on the board, or None if outside it."""
    if not (g.x0 <= px < g.x0 + g.cell * g.cols and g.y0 <= py < g.y0 + g.cell * g.rows):
        return None
    return int((py - g.y0) // g.cell), int((px - g.x0) // g.cell)


class App:
    def __init__(self):
        self.board_name = "TTT"
        self.x_kind = "Human"
        self.o_kind = "Minimax"
        self.depth = 3
        self.warning = ""
        self.buttons = []           # rebuilt each frame: (rect, callback)
        self.new_game()

    # --- config / setup ------------------------------------------------------

    def board_key(self):
        return BOARDS[self.board_name][0]

    def dims(self):
        return BOARDS[self.board_name][2]

    def new_game(self):
        self.warning = ""
        self.game = BOARDS[self.board_name][1]()
        self.agents = {P1: self._agent(self.x_kind), P2: self._agent(self.o_kind)}
        self.last_ai = pygame.time.get_ticks()

    def _agent(self, kind):
        if kind == "Human":
            return None                              # humans move via clicks
        if kind == "Random":
            return RandomAgent()
        if kind == "Minimax":
            depth = None if self.board_key() == "ttt" else self.depth
            return MinimaxAgent(depth=depth, seed=1)  # seeded -> varied lines to watch
        if kind == "Q-learn":
            path = f"qtable_{self.board_key()}.pkl.gz"
            if os.path.exists(path):
                q, meta = load_table(path)
                return QLearningAgent(epsilon=0.0, q=q, fold=meta.get("fold", False))
            self.warning = f"No {path} - train it first. Using Random instead."
            return RandomAgent()
        return RandomAgent()

    def cycle(self, attr, options):
        cur = getattr(self, attr)
        setattr(self, attr, options[(options.index(cur) + 1) % len(options)])
        self.new_game()

    def bump_depth(self, delta):
        self.depth = max(1, min(9, self.depth + delta))
        self.new_game()

    # --- geometry ------------------------------------------------------------

    def geometry(self):
        rows, cols = self.dims()
        avail_h = HEIGHT - PANEL_H
        cell = min(WIDTH // cols, avail_h // rows)
        x0 = (WIDTH - cell * cols) // 2
        y0 = PANEL_H + (avail_h - cell * rows) // 2
        return Geom(x0, y0, cell, rows, cols)

    # --- input ---------------------------------------------------------------

    def on_click(self, px, py):
        for rect, callback in self.buttons:            # controls take priority
            if rect.collidepoint(px, py):
                callback()
                return
        self._human_move(px, py)

    def _human_move(self, px, py):
        if self.game.is_terminal():
            return
        if self.agents[self.game.current_player()] is not None:
            return                                     # not a human's turn
        cell = pixel_to_cell(self.geometry(), px, py)
        if cell is None:
            return
        row, col = cell
        # tic-tac-toe move = cell index; Connect-4 move = column (gravity handles the row)
        move = row * self.dims()[1] + col if self.board_key() == "ttt" else col
        if move in self.game.legal_moves():
            self.game = self.game.apply(move)

    def step_ai(self):
        if self.game.is_terminal():
            return
        agent = self.agents[self.game.current_player()]
        if agent is None:
            return
        if pygame.time.get_ticks() - self.last_ai >= AI_DELAY_MS:
            self.game = self.game.apply(agent.choose(self.game))
            self.last_ai = pygame.time.get_ticks()

    # --- drawing -------------------------------------------------------------

    def draw(self, surf, font, small):
        surf.fill(BG)
        self._draw_controls(surf, small)
        self._draw_board(surf)
        self._draw_status(surf, font)

    def _button(self, surf, small, x, y, w, h, label, callback):
        rect = pygame.Rect(x, y, w, h)
        mouse = pygame.mouse.get_pos()
        pygame.draw.rect(surf, BTN_HI if rect.collidepoint(mouse) else BTN, rect, border_radius=6)
        text = small.render(label, True, TEXT)
        surf.blit(text, text.get_rect(center=rect.center))
        self.buttons.append((rect, callback))

    def _draw_controls(self, surf, small):
        self.buttons = []
        pygame.draw.rect(surf, PANEL, (0, 0, WIDTH, PANEL_H))
        # row 1: board + the two players
        self._button(surf, small, 12, 12, 96, 32, f"Board: {self.board_name}",
                     lambda: self.cycle("board_name", BOARD_ORDER))
        self._button(surf, small, 118, 12, 150, 32, f"Red: {self.x_kind}",
                     lambda: self.cycle("x_kind", PLAYER_KINDS))
        self._button(surf, small, 278, 12, 150, 32, f"Yellow: {self.o_kind}",
                     lambda: self.cycle("o_kind", PLAYER_KINDS))
        # row 2: minimax depth and restart
        self._button(surf, small, 12, 52, 32, 32, "-", lambda: self.bump_depth(-1))
        depth_label = small.render(f"depth {self.depth}", True, TEXT)
        surf.blit(depth_label, (52, 60))
        self._button(surf, small, 132, 52, 32, 32, "+", lambda: self.bump_depth(1))
        self._button(surf, small, 180, 52, 96, 32, "Restart", self.new_game)

    def _draw_board(self, surf):
        g = self.geometry()
        board = self.game.state_key()[0]
        for r in range(g.rows):
            for c in range(g.cols):
                x, y = g.x0 + c * g.cell, g.y0 + r * g.cell
                pygame.draw.rect(surf, PANEL, (x, y, g.cell, g.cell))
                pygame.draw.rect(surf, GRID, (x, y, g.cell, g.cell), 1)
                val = board[r * g.cols + c]
                color = DISC.get(val, EMPTY)
                pygame.draw.circle(surf, color, (x + g.cell // 2, y + g.cell // 2),
                                   g.cell // 2 - 6)

    def _draw_status(self, surf, font):
        if self.game.is_terminal():
            w = self.game.winner()
            msg = "Draw" if w is None else ("Red wins" if w == P1 else "Yellow wins")
        else:
            if self.game.current_player() == P1:
                who, kind = "Red", self.x_kind
            else:
                who, kind = "Yellow", self.o_kind
            msg = f"{who} to move  ({kind})"
        surf.blit(font.render(msg, True, TEXT), (12, HEIGHT - 30))
        if self.warning:
            surf.blit(font.render(self.warning, True, DISC[P1]), (12, HEIGHT - 58))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Learn-Vs-Search")
    font = pygame.font.SysFont(None, 26)
    small = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    app = App()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                app.on_click(*event.pos)
        app.step_ai()
        app.draw(screen, font, small)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
