import torch
import torch.nn.functional as F
from resnet import INPUT_SHAPE, MOVE_SHAPE
#TODO helper functions
def clean_string_board(board):
  board = board.replace('\n', '')
  board_arr = []
  for row in range(INPUT_SHAPE[1]+1):
    temp = []
    for col in range(INPUT_SHAPE[2]+1):
      temp.append(board[row*(INPUT_SHAPE[1]+1)+col])
    board_arr.append(temp)
  board_arr = board_arr[:-1]
  board = [row[1:] for row in board_arr] #get rid of row and column labels
  board.reverse()
  return board
'''
assuming board is represented in the form of str(state) from openspiel
'''
def board_to_input(board, turn, device):
  board = clean_string_board(board)
  res = torch.zeros(INPUT_SHAPE)
  for row in range(8):
    for col in range(8):
      if board[row][col] == 'o':
        res[0][row][col] = 1
      elif board[row][col] == 'O': #TODO double check this and the other king
        res[1][row][col] = 1
      elif board[row][col] == '+':
        res[2][row][col] = 1
      elif board[row][col] == '*':
        res[3][row][col] = 1
  if turn == 1:
    res[4] = torch.ones(INPUT_SHAPE[1:])
  res = res.to(device)
  return res
def input_to_board():
  ...
def output_to_move(board, output):
  #knowing that it must be a legal move, we can forego many checks
  #assuming output is the index in the output array that is a 1
  direction = output // (INPUT_SHAPE[1]*INPUT_SHAPE[2])
  rem1 = output % (INPUT_SHAPE[1]*INPUT_SHAPE[2])
  row = rem1 // INPUT_SHAPE[2]
  col = rem1 % INPUT_SHAPE[2]
  row_chr = str(row + 1)
  col_chr = str(chr(col + ord('a')))
  clean_board = clean_string_board(board)
  dest_diff = 1
  if direction == 0:
    if clean_board[row + 1][col + 1] != '.':
      dest_diff = 2
    return col_chr+row_chr+str(chr(col+ord('a')+dest_diff))+str(row+dest_diff + 1)
  elif direction == 1:
    if clean_board[row - 1][col + 1] != '.':
      dest_diff = 2
    return col_chr+row_chr+str(chr(col+ord('a')+dest_diff))+str(row-dest_diff + 1)
  elif direction == 2:
    if clean_board[row - 1][col - 1] != '.':
      dest_diff = 2
    return col_chr+row_chr+str(chr(col+ord('a')-dest_diff))+str(row-dest_diff + 1)
  else:
    assert direction == 3, "error calculating direction"
    if clean_board[row + 1][col - 1] != '.':
      dest_diff = 2
    return col_chr+row_chr+str(chr(col+ord('a')-dest_diff))+str(row+dest_diff + 1)


def move_to_output():
  ...

def encode_board(board):
  ...
def decode_board(board):
  ...
def encode_node(root):
  ...
def decode_node(root):
  ...
# def encode_move():
#   ...
# def decode_move():
#   ...
def mask_and_softmax(actions_raw, state):
  assert len(actions_raw.shape) == 2, "expected shape (batch_dim, prod(MOVE_DIM))"
  #TODO handle 0 legal move case?
  actions_raw = actions_raw.reshape((-1, *MOVE_SHAPE))
  legal_moves = [state.action_to_string(legal_action) for legal_action in state.legal_actions(state.current_player())] #TODO ah, legal actions different per batch
  tupled_moves = [
    (
      ord(move[0]) - ord("a"),
      int(move[1])-1,
      ord(move[2]) - ord("a"),
      int(move[3])-1
    )
    for move in legal_moves
  ]
  mapped_moves = [
    (0, move[1], move[0]) if move[2] > move[0] and move[3] > move[1]
    else (1, move[1], move[0]) if move[2] > move[0] and move[3] < move[1]
    else (2, move[1], move[0]) if move[2] < move[0] and move[3] < move[1]
    else (3, move[1], move[0])
    for move in tupled_moves
  ] 
  directions, rows, cols = zip(*mapped_moves)
  res = torch.full_like(actions_raw, float('-inf'))
  res[:, directions, rows, cols] = actions_raw[:, directions, rows, cols]
  res = F.softmax(res.flatten(start_dim=1), dim=1).reshape_as(res) #TODO ensure there is a batch dimension
  return res


