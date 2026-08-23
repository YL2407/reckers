import pyspiel
import random
from util import board_to_input

game = pyspiel.load_game("checkers")
state = game.new_initial_state()
# print(board_to_input(str(state), state.current_player(), 'cpu'))
print(str(state))
# print(state.legal_actions_mask(state.current_player()))
# for i in range(len(state.legal_actions_mask(state.current_player()))):
#   print(state.action_to_string(i))
# while not state.is_terminal():
#   action = random.choice(state.legal_actions(state.current_player()))
#   print(state.action_to_string(action))
#   state.apply_action(action)
# returns = state.returns()
# for pid in range(game.num_players()):
#   print("Utility for player {} is {}".format(pid, returns[pid]))