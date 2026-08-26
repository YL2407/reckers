import math

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

def puct(root, c=2.0):
  if root.turn == 1:
    obj = -root.qs + c*root.probs*torch.sqrt(torch.full_like(root.visit_counts, torch.sum(root.visit_counts).item()))/(torch.ones_like(root.visit_counts)+root.visit_counts)
  else:
    obj = root.qs + c*root.probs*torch.sqrt(torch.full_like(root.visit_counts, torch.sum(root.visit_counts).item()))/(torch.ones_like(root.visit_counts)+root.visit_counts)
  obj = torch.where(root.probs > 0, obj, torch.full_like(obj, float('-inf')))
  return torch.argmax(obj).item()

def mcts(model: ResNet, root: StateNode, sims=50):
  #TODO batch leaf node computations somehow
  assert not root.state.is_terminal(), "no running MCTS from a terminal state"
  (actions_raw, value) = model(board_to_input(root.board, root.turn))
  value = value.item()
  action_probs = mask_and_softmax(actions_raw, root.state).cpu()
  root.value_estimate = value
  root.probs = action_probs[0]
  for sim in range(sims):
    history = []
    curr = root
    chosen_idx = puct(curr)
    history.append((curr, chosen_idx))
    while curr.children.get(chosen_idx) != None:
      curr.visit_counts[output_to_tuple(chosen_idx)]+=1
      curr = curr.children[chosen_idx]
      if curr.state.is_terminal():
        value = curr.state.returns()[0]
        break
      chosen_idx = puct(curr)
      history.append((curr, chosen_idx))
    if not curr.state.is_terminal():
      curr.visit_counts[output_to_tuple(chosen_idx)]+=1
      curr.children[chosen_idx] = StateNode(curr.state.child(curr.state.string_to_action(output_to_move(curr.board, chosen_idx))))
      if curr.children[chosen_idx].state.is_terminal():
        value = curr.children[chosen_idx].state.returns()[0]
      else:
        (actions_raw, value) = model(board_to_input(curr.children[chosen_idx].board, curr.children[chosen_idx].turn))
        value = value.item()
        action_probs = mask_and_softmax(actions_raw, curr.children[chosen_idx].state).cpu()
        curr.children[chosen_idx].probs = action_probs[0]
      curr.children[chosen_idx].value_estimate = value
    #backprop
    while len(history) > 0:
      (parent, action_idx) = history.pop()
      parent.qs[output_to_tuple(action_idx)] = parent.qs[output_to_tuple(action_idx)] + (value - parent.qs[output_to_tuple(action_idx)])/parent.visit_counts[output_to_tuple(action_idx)] #online average update
  return root


  

