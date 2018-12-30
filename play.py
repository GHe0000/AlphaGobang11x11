# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3
"""

from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_11 import PolicyValueNet

N = 11

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location_str = input("You move:")
            location_str = location_str.split(",")
            location_int = [int(location_str[i]) for i in range(len(location_str))]
            location = [location_int[1], abs(location_int[0] - (N - 1))]
            move = board.location_to_move(location, N)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def run():
    model_file = "models/gomoku_11x11_5000.model"

    try:
        board = Board(N)
        game = Game(board)

        best_policy = PolicyValueNet(N, N, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
        human = Human()

        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
