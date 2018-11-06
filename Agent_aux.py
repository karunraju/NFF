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

    self.tmax = PARAM.tmax
    self.episode_buffer = [[]] * PARAM.A2C_EPISODE_SIZE
    self.net = A2C(self.episode_buffer, self.an)

    cur_dir = os.getcwd()
    self.dump_dir = cur_dir + '/tmp_' + self.method + '_' + time.strftime("%Y%m%d-%H%M%S") + '/'
    # Create output directory
    if not os.path.exists(self.dump_dir):
      os.makedirs(self.dump_dir)
    self.train_file = open(self.dump_dir + 'train_rewards.txt', 'w')
    self.test_file = open(self.dump_dir + 'test_rewards.txt', 'w')

    self.curr_state = self.env.reset()
    self.train_rewards = []
    self.test_rewards = []
    self.steps = 0
    self.cum_reward = 0.0
    self.save_count = 0

  def generate_episode(self, tmax, render=False)
    for i in range(tmax):
      softmax, action = self.net.get_output(self.curr_state)
      next_state, reward, _, _ = self.env.step(action)
      if render:
        env.render()
      self.episode_buffer[i] = (self.curr_state, action, reward, next_state, softmax)
      self.curr_state = next_state

      self.steps += 1
      self.cum_reward += reward
      if self.steps % 100 == 0:
        self.plot_train_stats()

  def train(self):
    for i in range(self.training_time)
      self.net.set_train()
      self.generate_episode(self.tmax)
      self.net.train()
      self.save_count += 1

  def test(self, testing_steps=100, model_file=None):
    if model_file is not None:
      self.net.load_model(model_file)

    self.net.set_eval()
    cum_reward = 0.0
    for i in range(testing_steps):
      softmax, action = self.net.get_output(self.curr_state)
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
    cum_reward = cum_reward/float(self.log_time)
    train_rewards.append(cum_reward)
    self.train_file.write(str(cum_reward))
    self.train_file.write('\n')
    self.train_file.flush()
    self.cum_reward = 0.0
    print('Train Reward: %.4f' % (train_rewards[-1]))
    self.steps = 0

    x = list(range(len(train_rewards)))
    plt.plot(x, train_rewards, '-bo')
    plt.xlabel('Time')
    plt.ylabel('Average Reward')
    plt.title('Training Curve')
    plt.savefig(self.dump_dir + 'Training_Curve_' + self.method + '.png')
    plt.close()

    plot(self.dump_dir + self.method, train_rewards)

    if self.save_count > 0 and self.save_count % 30 == 0:
      self.net.save_model_weights(self.save_count, self.dump_dir)
