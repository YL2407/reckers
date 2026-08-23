from resnet import ResNet, MOVE_SHAPE
from util import *
import math

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
  obj = root.qs + c*root.probs*math.sqrt(torch.sum(root.visit_counts))/(torch.ones_like(root.visit_counts)+root.visit_counts)
  if root.turn == 1:
    obj = -obj
  return torch.argmax(obj)

def mcts(model: ResNet, root, device, sims=50):
  #TODO only on first iteration?
  for sim in range(sims):
    history = [] #TODO append and pop (stack of parent history for backprop of values) (we get to do that online average calculation)
    (actions_raw, value) = model(board_to_input(root.board, root.turn, device))
    action_probs = mask_and_softmax(actions_raw, root.state)
    root.probs = action_probs
    root.value_estimate = value
    #TODO while loop somewhere below here
    chosen_idx = puct(root)
    if root.children.get(chosen_idx) == None:
      root.children[chosen_idx] = StateNode(root.state.child(root.state.string_to_action(output_to_move(root.board, chosen_idx))))
    root.visit_counts[chosen_idx]+=1
    (actions_raw, value) = model(board_to_input(root.children[chosen_idx].board), root.children[chosen_idx].turn, device)
    action_probs = mask_and_softmax(actions_raw, root.children[chosen_idx].state)
    root.children[chosen_idx].probs = action_probs
    root.children[chosen_idx].value_estimate = value

  #TODO go until leaf, loop (number of sims(decrement and LOOP, not recurse, we want efficiency))
  

