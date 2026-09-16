"""Step 14: light checks for the Pygame UI's pure helpers.

A graphical UI isn't unit-tested by pixels, but its geometry helper is pure logic we can
check. Importing ui does NOT open a window (pygame.init lives inside main()).
"""

from ui import Geom, pixel_to_cell


def test_pixel_inside_maps_to_the_right_cell():
    g = Geom(x0=0, y0=0, cell=10, rows=5, cols=4)
    assert pixel_to_cell(g, 5, 5) == (0, 0)        # top-left cell
    assert pixel_to_cell(g, 15, 25) == (2, 1)      # row 2, col 1
    assert pixel_to_cell(g, 35, 45) == (4, 3)      # bottom-right cell


def test_pixel_outside_the_board_is_none():
    g = Geom(x0=10, y0=10, cell=10, rows=3, cols=3)
    assert pixel_to_cell(g, 0, 0) is None          # above/left of the board
    assert pixel_to_cell(g, 100, 100) is None      # below/right of the board
