import pyspiel
from resnet import ResNet
from util import *
import torch.nn.functional as F

#TODO remember mixed/half precision


# model = ResNet()

# game = pyspiel.load_game("checkers")
# state = game.new_initial_state()
# print(str(state))
# while not state.is_terminal():
#   action = ...#random.choice(state.legal_actions(state.current_player()))
#   print(state.action_to_string(action))
#   state.apply_action(action)
# returns = state.returns()
# for pid in range(game.num_players()):
#   print("Utility for player {} is {}".format(pid, returns[pid]))

