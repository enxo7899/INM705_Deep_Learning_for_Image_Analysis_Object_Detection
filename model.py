# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

class CNN_Block(nn.Module):
  def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
    super(CNN_Block, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) # If batchnorm layer(bn_act) is true, then bias is False
    self.bn = nn.BatchNorm2d(out_channels)
    self.leaky = nn.LeakyReLU(0.1)
    self.use_bn_act = bn_act

  def forward(self, x):
    if self.use_bn_act:
      return self.leaky(self.bn(self.conv(x)))
    else:
      return self.conv(x)


class Residual_Block(nn.Module):
  def __init__(self, channels, use_residual=True, num_repeats=1):
    super(Residual_Block, self).__init__()
    self.layers = nn.ModuleList() # Like regular python list, but is container for pytorch nn modules

    for repeat in range(num_repeats):
      self.layers += [
          nn.Sequential(
            CNN_Block(channels, channels//2, kernel_size=1),
            CNN_Block(channels//2, channels, kernel_size=3, padding=1)
          )
      ]

    self.use_residual = use_residual
    self.num_repeats = num_repeats

  def forward(self, x):
    for layer in self.layers:
      if self.use_residual:
        x = x + layer(x)
      else:
        x = layer(x)

    return x
