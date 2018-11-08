import sys, os, time, gym, math, nel
import numpy as np, copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import hyperparameters as PARAM
from ReplayMemory import ReplayMemory
from replay_buffer.replay_buffer import ReplayBuffer
from models.A2C import A2C
from canonical_plot import plot

class Agent_aux():
  def __init__(self, render=False):

    # Create an instance of the network itself, as well as the memory.
    # Here is also a good place to set environmental parameters,
    # as well as training parameters - number of episodes / iterations, etc.
    self.render = render
    if render:
      self.env = gym.make('NEL-render-v0')
    else:
      self.env = gym.make('NEL-v0')
    #self.test_env = gym.make('NEL-v0')
    self.an = self.env.action_space.n               # No. of actions in env
    self.training_time = PARAM.TRAINING_TIME        # Training Time
    self.method = 'A2C'
    self.test_curr_state = None
    self.log_time = 100.0
    self.test_time = 1000.0

    self.burn_in = PARAM.BURN_IN
    self.tmax = PARAM.A2C_EPISODE_SIZE_MAX
    self.tmin = PARAM.A2C_EPISODE_SIZE_MIN
    self.seq_len = PARAM.A2C_SEQUENCE_LENGTH
    self.replay_buffer = ReplayBuffer(PARAM.REPLAY_MEMORY_SIZE)
    self.episode_buffer = [[]] * self.tmax
    self.net = A2C(self.episode_buffer, self.replay_buffer)

    cur_dir = os.getcwd()
    self.dump_dir = cur_dir + '/tmp_' + self.method + '_' + time.strftime("%Y%m%d-%H%M%S") + '/'
    # Create output directory
    if not os.path.exists(self.dump_dir):
      os.makedirs(self.dump_dir)
    self.train_file = open(self.dump_dir + 'train_rewards.txt', 'w')
    self.test_file = open(self.dump_dir + 'test_rewards.txt', 'w')

    self.curr_state = self.env.reset()
    self.tong_count = 0
    self.curr_state = self.burn_in_memory(self.curr_state)
    self.train_rewards = []
    self.test_rewards = []
    self.steps = 0
    self.cum_reward = 0.0
    self.save_count = 0

  def generate_episode(self, tmax, render=False):
    for i in range(tmax):
      val, softmax, action = self.net.get_output([i-1], seq_len=self.seq_len, batch_size=1)
      next_state, reward, _, _ = self.env.step(action)
      if render:
        self.env.render()
      if reward == 20.0:
        self.tong_count += 1
      elif reward == 100.0:
        self.tong_count -= 1
      self.episode_buffer[i] = (self.curr_state, action, reward/100.0, next_state, softmax, self.tong_count, val)
      self.replay_buffer.add(self.curr_state, action, reward/100.0, next_state, 0, self.tong_count)
      self.curr_state = next_state

      self.steps += 1
      self.cum_reward += reward
      if self.steps % 100 == 0:
        self.plot_train_stats()

  def train(self):
    for i in range(self.training_time):
      self.net.set_train()
      episode_len = np.random.randint(self.tmin, self.tmax+1)
      self.generate_episode(episode_len, self.render)
      self.net.train(episode_len)
      self.save_count += 1

  def test(self, testing_steps=100, model_file=None):
    if model_file is not None:
      self.net.load_model(model_file)

    self.net.set_eval()
    cum_reward = 0.0
    for i in range(testing_steps):
      softmax, action = self.net.get_output(self.curr_state, i)
      _, reward, _, _ = self.test_env.step(action)
      cum_reward += reward

    self.test_reward.append(cum_reward)
    self.test_file.write(str(test_rewards[-1]))
    self.test_file.write('\n')
    self.test_file.flush()
    print('\nTest Reward: %.4f\n' % (test_rewards[-1]))
    test_steps = 0

    x = list(range(len(test_rewards)))
    plt.plot(x, self.test_rewards, '-bo')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Testing Curve')
    plt.savefig(self.dump_dir + 'Testing_Curve_' + self.method + '.png')
    plt.close()

  def plot_train_stats(self):
    self.cum_reward = self.cum_reward/float(self.log_time)
    self.train_rewards.append(self.cum_reward)
    self.train_file.write(str(self.cum_reward))
    self.train_file.write('\n')
    self.train_file.flush()
    self.cum_reward = 0.0
    print('Train Reward: %.4f' % (self.train_rewards[-1]))
    self.steps = 0

    x = list(range(len(self.train_rewards)))
    plt.plot(x, self.train_rewards, '-bo')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Training Curve')
    plt.savefig(self.dump_dir + 'Training_Curve_' + self.method + '.png')
    plt.close()

    plot(self.dump_dir + self.method, self.train_rewards)

    if self.save_count > 0 and self.save_count % 30 == 0:
      self.net.save_model_weights(self.save_count, self.dump_dir)

  def burn_in_memory(self, curr_state):
    # Initialize your replay memory with a burn_in number of episodes / transitions.
    cnt = 0
    while self.burn_in > cnt:
      action = self.env.action_space.sample()
      next_state, reward, _, _ = self.env.step(action)
      if reward == 20.0:
        self.tong_count += 1
      elif reward == 100.0:
        self.tong_count -= 1
      self.replay_buffer.add(curr_state, action, reward/100.0, next_state, 0, self.tong_count)
      curr_state = next_state

      cnt = cnt + 1
    return curr_state
