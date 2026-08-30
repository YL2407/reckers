import sys
from mcts import StateNode, sequential_mcts
from resnet import ResNet
from train import det_select_mcts_action, select_mcts_action
from util import *
import pyspiel
import pygame


def game_loop(model, game):
  state = game.new_initial_state()
  pygame.init()
  screen = pygame.display.set_mode((720, 640))
  clock = pygame.time.Clock()
  running = True
  board_size = (640, 640)

  sq_w = board_size[0] / 8
  sq_h = board_size[1] / 8

  dark_brown = (93, 47, 39)
  light_brown = (255,190,108)

  piece_selected = False
  piece_pos = None
  clicked = False
  click_pos = None
  val = 0.5
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:
            clicked = True
            click_pos = event.pos
    screen.fill(light_brown)

    for sq_y in range(8):
      if sq_y % 2 == 0:
        start_x = 1
      else:
        start_x = 0
      for sq_x in range(start_x, 8, 2):
        pygame.draw.rect(screen, dark_brown, pygame.Rect(sq_x*sq_w, sq_y*sq_h, sq_w, sq_h))
    #TODO eval bar
    board = clean_string_board(str(state))
    board.reverse()
    for sq_y in range(len(board)):
      for sq_x in range(len(board[sq_y])):
        if board[sq_y][sq_x] != '.':
          fill_colour = 'white'
          outline_colour = 'black'
          if board[sq_y][sq_x] == 'o' or board[sq_y][sq_x] == '8':
            fill_colour = 'white'
            outline_colour = 'black'
          elif board[sq_y][sq_x] == '+' or board[sq_y][sq_x] == '*':
            fill_colour = 'black'
            outline_colour = 'white'
          pygame.draw.circle(screen, fill_colour, (sq_x*sq_w + sq_w/2, sq_y*sq_h+sq_h/2), min(sq_w, sq_h) * 2/5)
          pygame.draw.circle(screen, outline_colour, (sq_x*sq_w + sq_w/2, sq_y*sq_h+sq_h/2), min(sq_w, sq_h) * 2/5, width=2)
          if board[sq_y][sq_x] == '8' or board[sq_y][sq_x] == '*':
            pygame.draw.circle(screen, outline_colour, (sq_x*sq_w + sq_w/2, sq_y*sq_h+sq_h/2), min(sq_w, sq_h) * 1/10)
    pygame.draw.rect(screen, 'black', pygame.Rect(board_size[0], 0, 720-board_size[0], (1-val)*640))
    pygame.draw.rect(screen, 'white', pygame.Rect(board_size[0], (1-val)*640, 640, val*640))
    if not state.is_terminal():
      if state.current_player() == 1:
        (_, val_pred) = model(board_to_input(state))
        val = (val_pred[0].item() + 1)/2
        root = StateNode(state)
        with torch.no_grad():
          root = sequential_mcts(model, root)
        # chosen_idx = select_mcts_action(root)
        chosen_idx = det_select_mcts_action(root)
        action = state.string_to_action(output_to_move(root.board, chosen_idx))
        state.apply_action(action)
      else:
        if piece_selected and clicked:
          piece_selected = False
          clicked = False
          target = (int(click_pos[1])//int(sq_h), int(click_pos[0])//int(sq_w))
          start_to_string = str(chr(ord('a') + piece_pos[1]))+str(8 - piece_pos[0])
          target_to_string = str(chr(ord('a') + target[1]))+str(8 - target[0])
          try:
            state.apply_action_with_legality_check(state.string_to_action(start_to_string+target_to_string))
            (_, val_pred) = model(board_to_input(state))
            # print(val_pred[0].item())
            val = (val_pred[0].item() + 1)/2
          except:
            print("illegal move")
        elif clicked:
          if not click_pos[0] >= 640 and not click_pos[1] >= 640:
            clicked = False
            piece = board[int(click_pos[1])//int(sq_h)][int(click_pos[0])//int(sq_w)]
            # print(piece)
            if piece == 'o' or piece == '8':
              piece_selected = True
              piece_pos = (click_pos[1]//int(sq_h), click_pos[0]//int(sq_w))
    else:
      if clicked:
        print(state.returns()[0])
        clicked = False
    pygame.display.flip()
    clock.tick(60)

if __name__ == "__main__":
  weight_path = "weights/checkpoint_20.pth"
  if len(sys.argv) == 2:
    weight_path = sys.argv[1]
  model = ResNet()#.to(DEVICE)
  checkpoint = torch.load(weight_path, map_location=torch.device(DEVICE))
  model.load_state_dict(checkpoint["model"])
  game = pyspiel.load_game("checkers")
  game_loop(model, game)