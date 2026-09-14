"""Step 8 validation: the learner, tested against the PERFECT player.

Beating a random agent is easy; the decisive check is against minimax. Tic-tac-toe is
solved, so perfect play can't be beaten - which means a correctly-trained Q-agent should
STOP LOSING (drawing, not winning, is the ceiling). If the two-player credit assignment
were wrong, minimax would keep punishing it and losses would stay high. So "training
slashes losses against minimax" is exactly the signal that the learning is correct.

This test trains a small agent, so it takes a few seconds.
"""

from games.tictactoe import TicTacToe
from agents.qlearn import QLearningAgent, self_play_train, evaluate
from agents.minimax import MinimaxAgent


def ttt():
    return TicTacToe()


def test_training_stops_the_agent_losing_to_minimax():
    trained = QLearningAgent(alpha=0.2, epsilon=0.2, seed=0)
    self_play_train(trained, ttt, games=100000)
    untrained = QLearningAgent(epsilon=0.0, seed=0)

    games = 50
    _, _, trained_losses = evaluate(trained, MinimaxAgent(seed=1), ttt, games=games)
    _, _, untrained_losses = evaluate(untrained, MinimaxAgent(seed=1), ttt, games=games)

    # You can never BEAT perfect play, so success = avoiding losses. Training must cut
    # losses far below an untrained table's.
    assert trained_losses < untrained_losses
    assert trained_losses <= 0.2 * games      # loses at most ~10% vs the perfect player


def test_two_minimax_players_still_draw_with_randomized_tie_breaking():
    # Sanity check the new randomized minimax is still perfect: two of them must draw.
    game = ttt()
    a, b = MinimaxAgent(seed=1), MinimaxAgent(seed=2)
    players = {1: a, 2: b}
    while not game.is_terminal():
        game = game.apply(players[game.current_player()].choose(game))
    assert game.winner() is None
