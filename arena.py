import torch
from mcts import StateNode, sequential_mcts
from resnet import ResNet
import sys
from train import select_mcts_action, tuned_select_mcts_action, det_select_mcts_action
from util import *
import pyspiel

TEST_GAMES = 20


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#TODO revert to old sequential mcts when playing games

if __name__ == "__main__":
  model1_wins = 0
  draws = 0
  model2_wins = 0
  # model1 = ResNet().to(DEVICE)
  # model2 = ResNet().to(DEVICE)
  model1 = ResNet()
  model2 = ResNet()
  assert len(sys.argv) == 3, "provide 2 checkpoint file names as arguments"
  _, file1, file2 = sys.argv
  checkpoint1 = torch.load(file1, map_location=torch.device(DEVICE))
  checkpoint2 = torch.load(file2, map_location=torch.device(DEVICE))
  model1.load_state_dict(checkpoint1["model"])
  model1.eval()
  model2.load_state_dict(checkpoint2["model"])
  model2.eval()
  game = pyspiel.load_game("checkers")
  for game_iter in range(TEST_GAMES):
    print(f"game: {game_iter+1}")
    state = game.new_initial_state()
    while not state.is_terminal():
      root = StateNode(state)
      with torch.no_grad():
        if state.current_player() % 2 == game_iter % 2:
          root = sequential_mcts(model1, root)
          # chosen_idx = det_select_mcts_action(root)
        else:
          root = sequential_mcts(model2, root)
      chosen_idx = select_mcts_action(root)
      action = state.string_to_action(output_to_move(root.board, chosen_idx))
      state.apply_action(action)
    final_outcome = state.returns()[0]
    if final_outcome == 0:
      draws+=1
      print("draw")
    elif final_outcome == 1 and game_iter%2 == 0 or final_outcome == -1 and game_iter%2 == 1:
      model1_wins += 1
      print("model 1 win")
    else:
      model2_wins += 1
      print("model 2 win")
  print(f"model 1: {model1_wins} / draw: {draws} / model 2: {model2_wins}")

