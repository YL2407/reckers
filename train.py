import pyspiel
from resnet import ResNet
from util import *

#TODO remember mixed/half precision

class StateNode():
  def __init__(self, board):
    self.board = board
    self.N = 0
    self.Q = 0
    self.P = 0 #TODO this may change
    self.children = []

model = ResNet()

game = pyspiel.load_game("checkers")
state = game.new_initial_state()
print(str(state))
while not state.is_terminal():
  action = ...#random.choice(state.legal_actions(state.current_player()))
  print(state.action_to_string(action))
  state.apply_action(action)
returns = state.returns()
for pid in range(game.num_players()):
  print("Utility for player {} is {}".format(pid, returns[pid]))

