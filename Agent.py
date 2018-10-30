import sys, os, time, gym, math, nel
import numpy as np, copy
import matplotlib.pyplot as plt
from timeit import default_timer

import torch
import torch.nn as nn
import torch.optim as optim

from ReplayMemory import ReplayMemory
from models.Duel import DuelQNetwork
from models.Double_DQN import DoubleQNetwork

class Agent():
  def __init__(self, render=False, method='Duel', memory_size=100000):

    # Create an instance of the network itself, as well as the memory.
    # Here is also a good place to set environmental parameters,
    # as well as training parameters - number of episodes / iterations, etc.
    self.render = render
    if render:
      self.env = gym.make('NEL-render-v0')
      self.test_env = gym.make('NEL-render-v0') # Test environment
    else:
      self.env = gym.make('NEL-v0')
      self.test_env = gym.make('NEL-v0')
    self.an = self.env.action_space.n   # No. of actions in env
    self.epsilon = 0.5
    self.training_time = 3000000        # Training Time
    self.df = 0.9                       # Discount Factor
    self.batch_size = 32
    self.method = method
    self.test_curr_state = None
    self.log_time = 100.0
    self.test_time = 1000.0

    # Create Replay Memory and initialize with burn_in transitions
    self.exp_buff = ReplayMemory(memory_size=memory_size)
    self.burn_in_memory()

    # Create QNetwork instance
    if self.method == 'Duel':
      print('Using Duel Network.')
      self.net = DuelQNetwork(self.an)
    elif self.method == 'DoubleQ':
      print('Using DoubleQ Network.')
      self.net = DoubleQNetwork(self.an)
    else:
      raise NotImplementedError

    cur_dir = os.getcwd()
    self.dump_dir = cur_dir + '/tmp_' + self.method + '_' + time.strftime("%Y%m%d-%H%M%S") + '/'
    # Create output directory
    if not os.path.exists(self.dump_dir):
      os.makedirs(self.dump_dir)

  def update_epsilon(self):
    ''' Epsilon decay from 0.5 to 0.05 over 100000 iterations. '''
    if self.epsilon <= 0.05:
      self.epsilon = 0.05
      return

    self.epsilon = self.epsilon - (0.5 - 0.1)/10000000.0

  def epsilon_greedy_policy(self, q_values, epsilon):
    # Creating epsilon greedy probabilities to sample from.
    val = np.random.rand(1)
    if val <= epsilon:
      return np.random.randint(q_values.shape[0])
    return np.argmax(q_values)

  def greedy_policy(self, q_values):
    # Creating greedy policy for test time.
    return np.argmax(q_values)

  def train(self):
    dump_epochs = 100
    train_rewards = []
    test_rewards = []
    count = 0

    cum_reward = 0.0
    elapsed = 0.0
    curr_state = self.env.reset()
    if self.render:
      self.env.render()
    start_time = default_timer()
    test_start_time = default_timer()
    for i in range(self.training_time):
      # Get q_values based on the current state
      Vt, St = self.get_input_tensor(curr_state)
      q_values = self.net.get_Q_output(Vt, St)

      # Selecting an action based on the policy
      action = self.epsilon_greedy_policy(q_values, self.epsilon)

      # Executing action in simulator
      nextstate, reward, _, _ = self.env.step(action)
      if self.render:
        self.env.render()

      # Store Transition
      self.exp_buff.append((curr_state, action, reward/100, nextstate))

      # Sample random minibatch from experience replay
      batch = self.exp_buff.sample_batch(self.batch_size)

      # Train the Network with mini batches
      xVT, xST = self.get_input_tensors(batch)
      yT = self.get_output_tensors(batch)

      # Mask to select the actions from the Q network output
      mT = torch.zeros(self.batch_size, self.an, dtype=torch.uint8)
      for k, tran in enumerate(batch):
        mT[k, tran[1]] = 1
      self.net.train(xVT, xST, yT, mT)

      # Decay epsilon
      self.update_epsilon()

      cum_reward += reward
      curr_state = nextstate

      if default_timer() - start_time > self.log_time:
        cum_reward = cum_reward/float(self.log_time)
        elapsed += default_timer() - start_time
        start_time = default_timer()
        train_rewards.append(cum_reward*100)
        cum_reward = 0.0
        print('Elapsed Time:%.4f Train Reward: %.4f' % (elapsed, train_rewards[-1]))

        x = list(range(len(train_rewards)))
        plt.plot(x, train_rewards, '-bo')
        plt.xlabel('Time')
        plt.ylabel('Average Reward')
        plt.title('Training Curve')
        plt.savefig(self.dump_dir + 'Training_Curve_' + self.method + '.png')
        plt.close()


      if default_timer() - test_start_time > self.test_time:
        self.net.set_eval()
        test_rewards.append(self.test())
        test_start_time = default_timer()
        start_time = default_timer()            # Resetting train time after test
        self.net.set_train()
        count = count + 1
        print('\nElapsed Time:%.4f Test Reward: %.4f\n' % (elapsed, test_rewards[-1]))

        x = list(range(len(test_rewards)))
        plt.plot(x, test_rewards, '-bo')
        plt.xlabel('Time')
        plt.ylabel('Average Reward')
        plt.title('Testing Curve')
        plt.savefig(self.dump_dir + 'Testing_Curve_' + self.method + '.png')
        plt.close()

      if count > 0 and count % 10 == 0:
        self.net.save_model_weights(count, self.dump_dir)


  def test(self, testing_time=100, model_file=None, capture=False):
    if model_file is not None:
      self.net.load_model(model_file)

    if capture:
      self.test_env = gym.wrappers.Monitor(self.test_env, './')

    epsilon = 0.05
    rewards = []

    self.test_curr_state = self.test_env.reset()
    if self.render:
      self.test_env.render()
    cum_reward = 0.0
    for i in range(testing_time):
      # Initializing the episodes
      Vt, St = self.get_input_tensor(self.test_curr_state)
      q_values = self.net.get_Q_output(Vt, St)
      action = self.epsilon_greedy_policy(q_values, epsilon)

      # Executing action in simulator
      nextstate, reward, _, _ = self.test_env.step(action)
      if self.render:
        self.test_env.render()

      cum_reward += reward
      self.test_curr_state = nextstate
    avg_reward = cum_reward/float(testing_time)
    rewards.append(avg_reward)

    return avg_reward

  def burn_in_memory(self):
    # Initialize your replay memory with a burn_in number of episodes / transitions.
    cnt = 0
    burn_in = self.exp_buff.get_burn_in()
    while burn_in > cnt:
      curr_state = self.env.reset()
      while burn_in > cnt:
        # Randomly selecting action for burn in. Not sure if this is correct.
        action = self.env.action_space.sample()
        next_state, reward, _, _ = self.env.step(action)

        self.exp_buff.append((curr_state, action, reward/100, next_state))

        curr_state = next_state
        cnt = cnt + 1

  def get_input_tensor(self, obs):
    ''' Returns an input tensor from the observation. '''
    iV = np.zeros((1, 3, 11, 11))
    iS = np.zeros((1, 4))

    iV[0] = np.moveaxis(obs['vision'], -1, 0)
    iS[0] = np.concatenate((obs['scent'], np.array([int(obs['moved'])])), axis=0)
    iVt, iSt = torch.from_numpy(iV).float(), torch.from_numpy(iS).float()
    return iVt, iSt

  def get_input_tensors(self, batch, next_state=False):
    ''' Returns an input tensor created from the sampled batch. '''
    V = np.zeros((self.batch_size, 3, 11, 11))
    S = np.zeros((self.batch_size, 4))
    for i, tran in enumerate(batch):
      if next_state:
        obs = tran[3]                    # next state
      else:
        obs = tran[0]                    # current state

      V[i] = np.moveaxis(obs['vision'], -1, 0)
      S[i] = np.concatenate((obs['scent'], np.array([int(obs['moved'])])), axis=0)
    Vt, St = torch.from_numpy(V).float(), torch.from_numpy(S).float()
    return Vt, St

  def get_output_tensors(self, batch):
    ''' Returns an output tensor created from the sampled batch. '''
    Y = np.zeros(self.batch_size)
    Vt, St = self.get_input_tensors(batch, next_state=True)
    q_values_a = self.net.get_Q_output(Vt, St)
    q_values_e = self.net.get_target_output(Vt, St)
    for i, tran in enumerate(batch):
      action = self.greedy_policy(q_values_a[i])
      Y[i] = tran[2] + self.df*q_values_e[i][action]

    Yt = torch.from_numpy(Y).float()
    return Yt
