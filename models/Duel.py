import torch
import torch.nn as nn
import torch.optim as optim

import hyperparameters as PARAM
from nets.ResNet import ResNet

class DuelNet(nn.Module):
  def __init__(self, sz):
    super(DuelNet, self).__init__()
    self.res = ResNet()
    self.act = nn.ReLU()
    self.l1 = nn.Linear(256+4, 512)
    self.A_final_layer = nn.Linear(512, sz)
    self.V_final_layer = nn.Linear(512, 1)

  def forward(self, im, s):
    x = self.res(im)
    x = torch.cat((x, s), dim=1)
    x = self.act(self.l1(x))

    x_A = self.A_final_layer(x)
    x_V = self.V_final_layer(x)
    return x_V.expand(x_A.size()) + x_A - torch.mean(x_A)


class DuelQNetwork():
  def __init__(self, out_size):
    self.C = PARAM.C                  # Clone the Q network to target network for every C steps
    self.step_cnt = 0
    self.lr = PARAM.LEARNING_RATE     # Learning Rate

    # Q network and target network
    self.Q = DuelNet(out_size)
    self.Qt = DuelNet(out_size)

    # GPU availability
    self.gpu = torch.cuda.is_available()
    if self.gpu:
      print("Using GPU")
      self.Q = self.Q.cuda()
      self.Qt = self.Qt.cuda()
    else:
      print("Using CPU")

    # Update Qt weights with the weights of Q
    self.update_target_network()
    self.Qt.eval()

    # Loss Function and Optimizer
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr, weight_decay=1e-6)

    # Running Loss
    self.running_loss = 0.0

  def reduce_learning_rate(self):
    for pgroups in self.optimizer.param_groups:
      pgroups['lr'] = pgroups['lr']/10.0

  def train(self, XVt, XSt, Yt, mT):
    if self.gpu:
        XVt, XSt, Yt, mT = XVt.cuda(), XSt.cuda(), Yt.cuda(), mT.cuda()

    # zero the gradients
    self.optimizer.zero_grad()

    # forward + backward + optimize
    bO = torch.masked_select(self.Q(XVt, XSt), mT)
    loss = self.criterion(bO, Yt)
    loss.backward()
    self.optimizer.step()
    self.running_loss += loss.item()

    td_errors = (Yt - bO).clone().detach().cpu().numpy()

    # Update Target Network for every 'C' steps
    self.step_cnt = self.step_cnt + 1
    if self.step_cnt == self.C:
      self.update_target_network()
      self.step_cnt = 0
      self.running_loss = 0.0

    return td_errors

  def get_Q_output(self, Vt, St):
    ''' Returns output from the Q network. '''
    with torch.no_grad():
      if self.gpu:
        Vt, St = Vt.cuda(), St.cuda()
      Yt = self.Q(Vt, St)
      if self.gpu:
        Yt = Yt.cpu()
    return Yt.numpy()

  def get_target_output(self, Vt, St):
    ''' Returns output from the target Qt network. '''
    with torch.no_grad():
      if self.gpu:
        Vt, St = Vt.cuda(), St.cuda()
      Yt = self.Qt(Vt, St)
      if self.gpu:
        Yt = Yt.cpu()
    return Yt.numpy()

  def set_train(self):
    self.Q.train()

  def set_eval(self):
    self.Q.eval()

  def update_target_network(self):
    self.Qt.load_state_dict(self.Q.state_dict())

  def save_model_weights(self, suffix, path='./'):
    # Helper function to save your model / weights.
    state = {
              'epoch': suffix,
              'state_dict': self.Q.state_dict(),
              'optimizer': self.optimizer.state_dict()
            }
    torch.save(state, path + str(suffix) + '.dat')

  def load_model(self, model_file):
    # Helper function to load an existing model.
    state = torch.load(model_file)
    self.Q.load_state_dict(state['state_dict'])
    self.optimizer.load_state_dict(state['optimizer'])
    self.update_target_network()

