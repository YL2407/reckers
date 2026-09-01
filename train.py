import pyspiel
from resnet import ResNet
from util import *
import torch.nn.functional as F
import random
from mcts import mcts, StateNode
import multiprocessing
import time

GAMES_TO_SIM = 50
BATCH_SIZE = 32
MAX_TRAIN_ITERS = 1000
BUFFER_SIZE = 500000
TRAIN_LOOP_ITERS = 4000

replay_buffer = []  

def det_select_mcts_action(root):
  return torch.argmax(root.visit_counts).item()

def tuned_select_mcts_action(root, determinism = 1.0, diff_thresh = 0.3):
  counts = root.visit_counts.flatten()
  probs = counts / counts.sum()
  probs = torch.pow(probs, determinism)
  probs = probs / probs.sum()
  chosen_idx = torch.multinomial(probs, 1).item()
  if torch.abs(root.qs[output_to_tuple(torch.argmax(counts).item())]) > torch.abs(root.qs[output_to_tuple(chosen_idx)]) + diff_thresh:
    return tuned_select_mcts_action(root, determinism+0.5)
  return chosen_idx

def select_mcts_action(root):
  counts = root.visit_counts.flatten()
  probs = counts / counts.sum()
  chosen_idx = torch.multinomial(probs, 1).item()
  return chosen_idx
def self_play_game(game, model):
  game_state_array = []
  state = game.new_initial_state()
  while not state.is_terminal():
    root = StateNode(state)
    with torch.no_grad():
      # t1 = time.time()
      root = mcts(model, root)
      # t2 = time.time()
      # print(f"{64/(t2 - t1)} sims per second")
    game_state_array.append(encode_node(root))
    chosen_idx = select_mcts_action(root)
    action = state.string_to_action(output_to_move(root.board, chosen_idx))
    state = state.child(action)
  final_outcome = state.returns()[0]
  for game_state in game_state_array:
    replay_buffer.append((game_state, final_outcome))

def populate_replay_buffer(game, model):
  for iter in range(GAMES_TO_SIM//6):
    self_play_game(game, model)

def training_loop():
  model = ResNet().to(DEVICE)
  game = pyspiel.load_game("checkers")
  #TODO save weights from time to time, print iteration
  optim = torch.optim.AdamW(model.parameters())
  loss_fn_val = torch.nn.MSELoss()
  loss_fn_pol = torch.nn.CrossEntropyLoss()
  checkpoint = torch.load("weights/checkpoint_120.pth", map_location=torch.device(DEVICE), weights_only=False)
  model.load_state_dict(checkpoint["model"])
  optim.load_state_dict(checkpoint["optimizer"])
  replay_buffer = checkpoint["buffer"]
  for train_loop_iter in range(checkpoint["iteration"]+1, TRAIN_LOOP_ITERS):
    print(f"train loop iteration: {train_loop_iter}")
    # model = model.to('cpu')
    model.eval()
    for iter in range(GAMES_TO_SIM):
      print(f"game {iter+1} started")
      self_play_game(game, model)
    # p = []
    # num_processes = 6
    # communication_queue = multiprocessing.Queue(maxsize=10000)
    # #TODO start multiprocessing pool!
    # for i in range(num_processes):
    #   process = multiprocessing.Process(target = populate_replay_buffer, args = (communication_queue, game, model))
    #   p.append(process)
    #   process.start()
    # for process in p:
    #   process.join()
    # while not communication_queue.empty():
    #     replay_buffer.append(communication_queue.get())
    # for game in range(GAMES_TO_SIM):
    #   self_play_game()
    if len(replay_buffer) > BUFFER_SIZE:
      del replay_buffer[:-int(BUFFER_SIZE/2)]
    # model = model.to(DEVICE)
    model.train()
    print("starting weight update")
    for batch in range(MAX_TRAIN_ITERS):
      #sample batch
      raw_batch = random.sample(replay_buffer, BATCH_SIZE)
      #map batch to usable data
      batch_val_y = torch.stack([torch.tensor(example[1]) for example in raw_batch]).reshape(BATCH_SIZE, -1).to(DEVICE)
      rem_vals = [decode_node(example[0]) for example in raw_batch]
      (batch_x, batch_pol_y) = zip(*rem_vals)
      batch_x = torch.cat(batch_x, dim=0).to(DEVICE)
      batch_pol_y = torch.stack(batch_pol_y).reshape(BATCH_SIZE, -1).to(DEVICE)
      #forward pass
      (pred_pol, pred_val) = model(batch_x)
      loss_pol = loss_fn_pol(pred_pol, batch_pol_y)
      loss_val = loss_fn_val(pred_val, batch_val_y)
      (loss_pol + loss_val).backward()
      optim.step()
      optim.zero_grad()
    if (train_loop_iter+1) % 20 == 0:
      print("saving weights...")
      torch.save({
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
        "iteration": train_loop_iter,
        "buffer": replay_buffer
      }, f"weights/checkpoint_{train_loop_iter+1}.pth")


if __name__ == "__main__":
  training_loop()

  # training_loop()
# print(str(state))
#   action = ...#random.choice(state.legal_actions(state.current_player()))
#   print(state.action_to_string(action))
#   state.apply_action(action)
# returns = state.returns()
# for pid in range(game.num_players()):
#   print("Utility for player {} is {}".format(pid, returns[pid]))

