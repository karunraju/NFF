import torch
import torch.nn as nn
import torch.optim as optim

import hyperparameters as PARAM
import aux.Multimodal as Multimodal

class A2C():
  def __init__(self, episode_buffer, state_size, action_space=3, N=100):
    self.lr = PARAM.LEARNING_RATE     # Learning Rate
    self.episode_buffer = episode_buffer
    self.tmax = PARAM.tmax
    self.N = PARAM.N

    # A2C network
    self.A = Multimodal.Multimodal(state_size, action_sapce)

    # GPU availability
    self.gpu = torch.cuda.is_available()
    if self.gpu:
      print("Using GPU")
      self.A = self.A.cuda()
    else:
      print("Using CPU")

    # Loss Function and Optimizer
    self.val_optim = optim.Adam(self.A.parameters(), lr=self.lr, weight_decay=1e-6)
    self.pol_optim = optim.Adam(self.A.parameters(), lr=self.lr, weight_decay=1e-6)

  def reduce_learning_rate(self):
    for pgroups in self.optimizer.param_groups:
      pgroups['lr'] = pgroups['lr']/10.0

  def train(self)
    self.val_optim.zero_grad()
    self.pol_optim.zero_grad()

    ploss, vloss = self.compute_loss()
    ploss.backward(retain_graph=True)
    vloss.backward()

    self.val_optim.step()
    self.pol_optim.step()

  def compute_loss(self):
    T = self.tmax
    for t in range(T-1, -1, -1):
      state, action, reward, next_state, softmax = self.episode_buffer[t]
      vt, st = self.get_input_tensor(tn_state)
      val = self.A.forward(vt, st)
      if t + n >= T:
        Vend = 0
      else:
        tn_state, _, _, _, _ = self.episode_buffer[t+n]
        vt, st = self.get_input_tensor(tn_state)
        Vend = self.A.forward(vt, st)
      sum_ = 0.0
      for k in range(n):
        if t + k < T:
          _, _, tk_reward, _, _ = self.episode_buffer[t+k]
          sum_ += tk_reward[t+k] * (gamma**k)
      rew = Vend*(gamma**n) + float(sum_)
      if t == T-1:
         ploss = (rew - val)*pt.log(softmax)
         vloss = (rew - val)**2
      else:
         ploss += (rew - val)*pt.log(softmax)
         vloss += (rew - val)**2

    ploss = -1.0*ploss/float(T)
    vloss = vloss/float(T)
    return ploss, vloss

  def get_output(self, state, no_grad=False):
    ''' Returns output from the A network. '''
    vt, st = self.get_input_tensor(state)
    if no_grad:
      with torch.no_grad():
        Yt = self.A.forward(Vt, St)
    else:
      Yt = self.A.forward(Vt, St)

    # TODO: Add softmax
    action = np.random.choice(np.arange(4), 1, p=sm.clone().detach().numpy())[0]
    return softmax, action

  def get_input_tensor(self, obs):
    ''' Returns an input tensor from the observation. '''
    iV = np.zeros((1, 3, 11, 11))
    iS = np.zeros((1, 4))

    iV[0] = np.moveaxis(obs['vision'], -1, 0)
    iS[0] = np.concatenate((obs['scent'], np.array([int(obs['moved'])])), axis=0)
    iVt, iSt = torch.from_numpy(iV).float(), torch.from_numpy(iS).float()

    if self.gpu:
      iVt, iSt = iVt.cuda(), iSt.cuda()
    return iVt, iSt

  def set_train(self):
    self.A.train()

  def set_eval(self):
    self.A.eval()

  def save_model_weights(self, suffix, path='./'):
    # Helper function to save your model / weights.
    state = {
              'epoch': suffix,
              'state_dict': self.A.state_dict(),
              'opt_val': self.opt_val.state_dict()
              'opt_pol': self.opt_pol.state_dict()
            }
    torch.save(state, path + str(suffix) + '.dat')

  def load_model(self, model_file):
    # Helper function to load an existing model.
    state = torch.load(model_file)
    self.A.load_state_dict(state['state_dict'])
    self.opt_val.load_state_dict(state['opt_val'])
    self.opt_pol.load_state_dict(state['opt_pol'])
