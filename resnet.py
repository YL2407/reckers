import torch
import torch.nn as nn
import torch.nn.functional as F
import math

INPUT_SHAPE = (5, 8, 8) #checkers board: w pawns, w kings, b pawns, b kings, turn
BLOCK_CHANNELS = 64
KERNEL_SIZE = 3
POLICY_CHANNELS = 2
MOVE_SHAPE = (4, 8, 8) #(top right, bottom right, bottom left, top left) for each square
VALUE_CHANNELS = 1
VALUE_HIDDEN_DIM = 64
NUM_BLOCKS = 4


class ResBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(BLOCK_CHANNELS, BLOCK_CHANNELS, KERNEL_SIZE, padding='same')
    self.bn1 = nn.BatchNorm2d(BLOCK_CHANNELS)
    self.conv2 = nn.Conv2d(BLOCK_CHANNELS, BLOCK_CHANNELS, KERNEL_SIZE, padding='same')
    self.bn2 = nn.BatchNorm2d(BLOCK_CHANNELS)
  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + residual
    out = F.relu(out)
    return out

class PolicyHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(BLOCK_CHANNELS, POLICY_CHANNELS, 1)
    self.fc1 = nn.Linear(POLICY_CHANNELS*INPUT_SHAPE[1]*INPUT_SHAPE[2], math.prod(MOVE_SHAPE))

  def forward(self, x):
    out = self.conv1(x)
    out = F.relu(out)
    out = torch.flatten(out, start_dim=1)
    out = self.fc1(out)
    return out #raw scores, requires softmax

class ValueHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(BLOCK_CHANNELS, VALUE_CHANNELS, 1)
    self.fc1 = nn.Linear(INPUT_SHAPE[1]*INPUT_SHAPE[2], VALUE_HIDDEN_DIM)
    self.fc2 = nn.Linear(VALUE_HIDDEN_DIM, 1)
  def forward(self, x):
    out = self.conv1(x)
    out = F.relu(out)
    out = torch.flatten(out, start_dim=1)
    out = self.fc1(out)
    out = F.relu(out)
    out = self.fc2(out)
    out = F.tanh(out)
    return out

class ResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(INPUT_SHAPE[0], BLOCK_CHANNELS, 3, padding='same')
    self.bn1 = nn.BatchNorm2d(BLOCK_CHANNELS)
    self.res_blocks = nn.ModuleList([
      ResBlock() for i in range(NUM_BLOCKS)
    ])
    self.policy_head = PolicyHead()
    self.value_head = ValueHead()
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    for i in range(NUM_BLOCKS):
      x = self.res_blocks[i](x)
    policy_out = self.policy_head(x)
    value_out = self.value_head(x)
    return (policy_out, value_out)
