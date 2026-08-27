import math
import time

from resnet import MOVE_SHAPE, ResNet
from util import *

class StateNode():
  def __init__(self, state):
    self.state = state
    self.board = str(state)
    self.turn = state.current_player()
    self.children = {}
    self.visit_counts = torch.zeros(MOVE_SHAPE) #TODO may change to dictionaries
    self.probs = torch.zeros(MOVE_SHAPE) #initialise these to the probabilities given by model
    self.qs = torch.zeros(MOVE_SHAPE)
    self.value_estimate = 0 #network's estimate of this state's value
    self.requires_eval = True

def puct(root, c=2.0):
  if root.turn == 1:
    obj = -root.qs + c*root.probs*torch.sqrt(torch.full_like(root.visit_counts, torch.sum(root.visit_counts).item()))/(torch.ones_like(root.visit_counts)+root.visit_counts)
  else:
    obj = root.qs + c*root.probs*torch.sqrt(torch.full_like(root.visit_counts, torch.sum(root.visit_counts).item()))/(torch.ones_like(root.visit_counts)+root.visit_counts)
  obj = torch.where(root.probs > 0, obj, torch.full_like(obj, float('-inf')))
  return torch.argmax(obj).item()

#TODO multithreading
def mcts(model: ResNet, root: StateNode, sims=50):
  #TODO batch leaf node computations somehow
  assert not root.state.is_terminal(), "no running MCTS from a terminal state"
  #model = model.to('cpu') #TODO definitely move to the GPU if batching ... (in that case, move the data too)
  # t1 = time.time()
  (actions_raw, value) = model(board_to_input(root.board, root.turn))
  # t2 = time.time()
  # print(f"time for initial model eval: {t2-t1}")
  value = value.item()
  action_probs = mask_and_softmax(actions_raw, root.state, root.state.legal_actions())#.cpu()
  root.value_estimate = value
  root.probs = action_probs[0]
  root.requires_eval = False
  pending_evals = []
  for sim in range(sims):
    evals = 0
    # t1 = time.time()
    history = []
    curr = root
    chosen_idx = puct(curr)
    history.append((curr, chosen_idx))
    while curr.children.get(chosen_idx) != None:
      curr.visit_counts[output_to_tuple(chosen_idx)]+=1
      curr = curr.children[chosen_idx]
      if curr.state.is_terminal():
        curr.requires_eval = False
        value = curr.state.returns()[0]
        break
      if not curr.requires_eval:
        chosen_idx = puct(curr)
        history.append((curr, chosen_idx))
      else:
        evals += len(pending_evals)
        input_batch = [board_to_input(pos[0].board, pos[0].turn) for pos in pending_evals]
        input_batch = torch.cat(input_batch, dim=0)
        (pol_batch, val_batch) = model(input_batch)
        for i in range(input_batch.shape[0]):
          action_probs = mask_and_softmax(pol_batch[i].unsqueeze(0), pending_evals[i][0].state, pending_evals[i][1])
          value = val_batch[i].item()
          pending_evals[i][0].probs = action_probs[0]
          pending_evals[i][0].value_estimate = value
          pending_evals[i][0].requires_eval = False
        pending_evals = []
        chosen_idx = puct(curr)
        history.append((curr, chosen_idx))
    if not curr.state.is_terminal():
      curr.visit_counts[output_to_tuple(chosen_idx)]+=1
      curr.children[chosen_idx] = StateNode(curr.state.child(curr.state.string_to_action(output_to_move(curr.board, chosen_idx))))
      if curr.children[chosen_idx].state.is_terminal():
        value = curr.children[chosen_idx].state.returns()[0]
        curr.children[chosen_idx].value_estimate = value
        curr.children[chosen_idx].requires_eval = False
      else:
        pending_evals.append((curr.children[chosen_idx], curr.children[chosen_idx].state.legal_actions()))
        # (actions_raw, value) = model(board_to_input(curr.children[chosen_idx].board, curr.children[chosen_idx].turn))
        # value = value.item()
        # action_probs = mask_and_softmax(actions_raw, curr.children[chosen_idx].state, curr.children[chosen_idx].state.legal_actions())#.cpu()
        # curr.children[chosen_idx].probs = action_probs[0]
        # curr.children[chosen_idx].value_estimate = value
    # if len(pending_evals) > 0:
    #   print("evaluating?")
    #   input_batch = [board_to_input(pos[0].board, pos[0].turn) for pos in pending_evals]
    #   input_batch = torch.cat(input_batch, dim=0)
    #   (pol_batch, val_batch) = model(input_batch)
    #   for i in range(input_batch.shape[0]):
    #     action_probs = mask_and_softmax(pol_batch[i].unsqueeze(0), pending_evals[i][0].state, pending_evals[i][1])
    #     value = val_batch[i].item()
    #     pending_evals[i][0].probs = action_probs[0]
    #     pending_evals[i][0].value_estimate = value
    #     pending_evals[i][0].requires_eval = False
    #   pending_evals = []
      # t2 = time.time()
      # print(f"time for single sim: {t2-t1}")
      #backprop
    while len(history) > 0: #TODO does backprop require the latest values?... does this defeat the purpose of batching?
      (parent, action_idx) = history.pop()
      parent.qs[output_to_tuple(action_idx)] = parent.qs[output_to_tuple(action_idx)] + (value - parent.qs[output_to_tuple(action_idx)])/parent.visit_counts[output_to_tuple(action_idx)] #online average update
  return root


  

