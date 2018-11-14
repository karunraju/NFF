import torch
import torch.nn as nn
import torch.optim as optim

class Residual(nn.Module):
  def __init__(self, ch):
    super(Residual, self).__init__()
    self.rl = nn.LeakyReLU(inplace=True)
    self.c1 = nn.Conv2d(ch, ch, 3, stride=1, padding=1)
    self.c2 = nn.Conv2d(ch, ch, 3, stride=1, padding=1)
    self.initialize()
  
  def initialize(self):
    nn.init.xavier_uniform_(self.c1.weight.data)
    nn.init.constant_(self.c1.bias.data, 0)
    nn.init.xavier_uniform_(self.c2.weight.data)
    nn.init.constant_(self.c2.bias.data, 0)

  def forward(self, x):
    out = self.c1(x)
    out = self.c2(self.rl(out))
    
    out += x
    out = self.rl(out)
    return out


class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()
    rl = nn.LeakyReLU(inplace=True)
    l1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
    b1 = Residual(32)
    l2 = nn.Conv2d(32, 64, 5, stride=1, padding=0)
    b2 = Residual(64)
    mp = nn.MaxPool2d(3)
    self.seq = nn.Sequential(l1, rl, b1, l2, rl, b2, mp)
    self.initialize()

  def initialize(self):
    for layer in self.seq:
      try:
        nn.init.xavier_uniform_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
      except:
        pass

  def forward(self, x):
    x = self.seq(x)
    x = x.view(x.size(0), -1)

    return x
