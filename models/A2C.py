import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import hyperparameters as PARAM
from aux.AuxNetwork import AuxNetwork
from models.Ensemble import Ensemble

class A2C():
  def __init__(self, ReplayBuffer, action_space=3, network=None):
    self.lr = PARAM.LEARNING_RATE
    self.N = PARAM.N
    self.gamma = PARAM.gamma
    self.seq_len = PARAM.A2C_SEQUENCE_LENGTH
    self.aux_batch_size = PARAM.AUX_TASK_BATCH_SIZE
    self.vfr_weight = PARAM.VFR_LOSS_WEIGHT
    self.rp_weight = PARAM.RP_LOSS_WEIGHT
    self.pc_weight = PARAM.PC_LOSS_WEIGHT
    self.gpu = torch.cuda.is_available()

    # A2C network
    if PARAM.ENSEMBLE<1:
      self.A = AuxNetwork(state_size=PARAM.STATE_SIZE, action_space=action_space, seq_len=self.seq_len)
      # GPU availability
      if self.gpu:
        print("Using GPU")
        self.A = self.A.cuda()
      else:
        print("Using CPU")
      self.replay_buffer = ReplayBuffer(PARAM.REPLAY_MEMORY_SIZE)
      # Loss Function and Optimizer
      self.optimizer = optim.Adam(self.A.parameters(), lr=self.lr, weight_decay=1e-6)
    else:
      self.Ensemble = Ensemble(PARAM.ENSEMBLE, action_space, self.seq_len, ReplayBuffer, network)
      self.source_context()

    self.vfr_criterion = nn.MSELoss()           # Value Function Replay loss
    self.rp_criterion = nn.CrossEntropyLoss()   # Reward Prediction loss
    self.pc_criterion = nn.MSELoss()            # Value Function Replay loss

  def add_episodic_buffer(self, episode_buffer):
    self.episode_buffer = episode_buffer

  def reduce_learning_rate(self):
    if PARAM.ENSEMBLE >= 1:
      self.Ensemble.reduce_learning_rate()
      return

    for pgroups in self.optimizer.param_groups:
      pgroups['lr'] = pgroups['lr']/10.0


  def source_context(self):
    if PARAM.ENSEMBLE==0:
      return
    self.Ensemble.update_context()
    self.A = self.Ensemble.get_network()
    self.optimizer = self.Ensemble.get_optimizer()
    self.replay_buffer = self.Ensemble.get_replay_buffer()

  def reduce_learning_rate(self):
    for pgroups in self.optimizer.param_groups:
      pgroups['lr'] = pgroups['lr']/10.0

  def train(self, episode_number, episode_len):
    self.optimizer.zero_grad()
    loss = self.compute_A2C_loss(episode_len)
    if episode_number>5:
      loss += self.vfr_weight * self.compute_vfr_loss()
      if self.replay_buffer.any_reward_instances():
        loss += self.rp_weight * self.compute_rp_loss()
      loss += self.pc_weight * self.compute_pc_loss()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.A.parameters(), PARAM.GRAD_CLIP_VAL)
    self.optimizer.step()

    if math.isnan(loss.item()):
      print('Loss Eploded!')

  def compute_A2C_loss(self, episode_len):
    T = episode_len
    n = self.N
    for t in range(T-1, -1, -1):
      val = self.episode_buffer[t][-1]
      if t + n >= T:
        Vend = 0
      else:
        Vend = self.episode_buffer[t+n][-1]
      sum_ = 0.0
      for k in range(n):
        if t + k < T:
          tk_reward = self.episode_buffer[t+k][2]
          sum_ += tk_reward * (self.gamma**k)
      rew = Vend*(self.gamma**n) + float(sum_)
      if t == T-1:
         ploss = (rew - val)*torch.log(self.episode_buffer[t][4][self.episode_buffer[t][1]])
         vloss = (rew - val)**2
      else:
         ploss += (rew - val)*torch.log(self.episode_buffer[t][4][self.episode_buffer[t][1]])
         vloss += (rew - val)**2

    ploss = -1.0*ploss/float(T)
    vloss = vloss/float(T)

    return ploss + vloss

  def compute_vfr_loss(self):
    """ Computes Value Function Replay Loss. """
    idxs = self.replay_buffer.sample_idxs(self.aux_batch_size)
    vision, scent, state, reward = self.get_io_from_replay_buffer(idxs, batch_size=self.aux_batch_size, seq_len=self.seq_len)
    val, _ = self.A.forward(vision, scent, state)

    return self.vfr_criterion(val.view(-1, 1), reward)

  def compute_rp_loss(self):
    """ Computes Reward Prediction Loss. """
    vision, ground_truth = self.get_io_from_skewed_replay_buffer(batch_size=self.aux_batch_size, seq_len=3)
    pred = self.A.predict_rewards(vision)

    return self.rp_criterion(pred, ground_truth)

  def compute_pc_loss(self):
    """ Computes Pixel Control Loss. """
    idxs = self.replay_buffer.sample_idxs(self.aux_batch_size)
    vision, aux_rew, actions = self.get_pc_io_from_replay_buffer(idxs, batch_size=self.aux_batch_size, seq_len=1)
    pred = self.A.pixel_control(vision)
    for i in range(20):
      if i == 0:
        pc_loss = self.pc_criterion(aux_rew[i], pred[i, actions[i]])
      else:
        pc_loss += self.pc_criterion(aux_rew[i], pred[i, actions[i]])

    return pc_loss

  def get_output(self, index, batch_size=1, seq_len=1, no_grad=False):
    ''' Returns output from the A network. '''
    vision, scent, state = self.get_input_tensor(index, batch_size, seq_len)
    if PARAM.ENSEMBLE!=0:
      self.A = self.Ensemble.get_network()
      self.optimizer = self.Ensemble.get_optimizer()
    if no_grad:
      with torch.no_grad():
        val, softmax = self.A.forward(vision, scent, state)
    else:
      val, softmax = self.A.forward(vision, scent, state)

    action = np.random.choice(np.arange(3), 1, p=np.squeeze(softmax.clone().cpu().detach().numpy()))
    return val, softmax.view(3), action

  def get_input_tensor(self, idxs, batch_size=1, seq_len=1):
    ''' Returns an input tensor from the observation. '''
    vision = np.zeros((batch_size, seq_len, 3, 11, 11))
    scent = np.zeros((batch_size, seq_len, 3))
    state = np.zeros((batch_size, seq_len, 4))

    for k, idx in enumerate(idxs):
      for j in range(seq_len):
        if idx - j < 0:
          continue
        obs, action, rew, _, _, tong_count, _ = self.episode_buffer[idx-j]
        vision[k, j] = np.moveaxis(obs['vision'], -1, 0)
        scent[k, j] = obs['scent']
        state[k, j] = np.array([action, rew, int(obs['moved']), tong_count])

    vision, scent, state = torch.from_numpy(vision).float(), torch.from_numpy(scent).float(), torch.from_numpy(state).float()
    if self.gpu:
      vision, scent, state = vision.cuda(), scent.cuda(), state.cuda()

    return vision, scent, state

  def get_io_from_replay_buffer(self, idxs, batch_size=1, seq_len=1):
    ''' Returns an input tensor from the observation. '''
    vision = np.zeros((batch_size, seq_len, 3, 11, 11))
    scent = np.zeros((batch_size, seq_len, 3))
    state = np.zeros((batch_size, seq_len, 4))
    reward = np.zeros((batch_size, 1))

    for k, idx in enumerate(idxs):
      for j in range(seq_len):
        obs, action, rew, _, _, tong_count = self.replay_buffer.get_single_sample(idx-j)
        vision[k, j] = np.moveaxis(obs['vision'], -1, 0)
        scent[k, j] = obs['scent']
        state[k, j] = np.array([action, rew, int(obs['moved']), tong_count])
        if j == 0:
          reward[k] = rew

    vision, scent, state, reward = torch.from_numpy(vision).float(), torch.from_numpy(scent).float(), torch.from_numpy(state).float(), torch.from_numpy(reward).float()
    if self.gpu:
      vision, scent, state, reward = vision.cuda(), scent.cuda(), state.cuda(), reward.cuda()

    return vision, scent, state, reward

  def get_io_from_skewed_replay_buffer(self, batch_size=1, seq_len=1):
    ''' Returns an input tensor from the observation. '''
    vision, reward_class = self.replay_buffer.skewed_samples(batch_size, seq_len)
    vision, reward_class = torch.from_numpy(vision).float(), torch.from_numpy(reward_class).long()
    if self.gpu:
      vision, reward_class = vision.cuda(), reward_class.cuda()

    return vision, reward_class

  def get_pc_io_from_replay_buffer(self, idxs, batch_size=1, seq_len=1):
    ''' Returns an input tensor from the observation. '''
    vision = np.zeros((batch_size, seq_len, 3, 11, 11))
    aux_rew = np.zeros((batch_size, 11, 11))
    actions = [[]]*batch_size

    for k, idx in enumerate(idxs):
      for j in range(seq_len):
        obs, action, _, next_obs, _, _ = self.replay_buffer.get_single_sample(idx-j)
        vision[k, j] = np.moveaxis(obs['vision'], -1, 0)
        if j == 0:
          if next_obs['moved']:
            aux_rew[k] = np.mean(np.abs(obs['vision'] - next_obs['vision']), axis=2)
          actions[k] = action

    vision, aux_rew = torch.from_numpy(vision).float(), torch.from_numpy(aux_rew).float()
    if self.gpu:
      vision, aux_rew = vision.cuda(), aux_rew.cuda()

    return vision, aux_rew, actions

  def set_train(self):
    self.A.train()

  def set_eval(self):
    self.A.eval()

  def save_model_weights(self, suffix, path='./'):
    # Helper function to save your model / weights.
    if PARAM.ENSEMBLE!=0:
      self.Ensemble.save()
      return
    state = {
              'epoch': suffix,
              'state_dict': self.A.state_dict(),
              'optmizer': self.optimizer.state_dict(),
            }
    torch.save(state, path + str(suffix) + '.dat')

  def load_model(self, model_file):
    # Helper function to load an existing model.
    if PARAM.ENSEMBLE != 0:
      print('Loading Ensemble models')
      self.Ensemble.load()
      return
    #state = torch.load(model_file)
    #self.A.load_state_dict(state['state_dict'])
    #self.optimizer.load_state_dict(state['optimizer'])
    self.A.load(model_file)

  def get_replay_buffer(self):
    if PARAM.ENSEMBLE!=0:
      return self.Ensemble.get_replay_buffer()
    return self.replay_buffer

  def get_action_repeat(self):
    if PARAM.ENSEMBLE!=0:
      return self.Ensemble.get_action_repeat()
    return PARAM.ACTION_REPEAT

  def monitor(self, rewards_list):
    if PARAM.ENSEMBLE==0:
      return
    self.Ensemble.analyze_rewards(rewards_list)
