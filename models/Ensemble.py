import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import hyperparameters as PARAM
from aux.AuxNetwork import AuxNetwork

class Ensemble():
	def __init__(self, size, action_space, seq_len, ReplayBuffer):
		self.list_of_networks = [AuxNetwork(state_size=PARAM.STATE_SIZE, action_space=action_space, seq_len=seq_len) for i in range(size)]
		self.gpu = torch.cuda.is_available()
		if self.gpu:
			print("Using GPU")
			self.list_of_networks = [network.cuda() for network in self.list_of_networks]
		else:
			print("Using CPU")
		self.list_of_optimizers = [optim.Adam(network.parameters(), lr=PARAM.LEARNING_RATE, weight_decay=1e-6) for network in self.list_of_networks]
		self.list_of_replay_buffers = [ReplayBuffer(PARAM.REPLAY_MEMORY_SIZE) for network in self.list_of_networks]
		self.list_of_action_repeats = PARAM.ACTION_REPEAT
		self.current=len(self.list_of_networks)-1
		self.update_context()

	def analyze_rewards(self, rewards_list):
		rewards = np.array(rewards_list)
		episode_length = rewards.shape[0]
		if rewards[:episode_length//2].sum()>2*rewards[episode_length//2:].sum() and rewards.mean()>2:		#First half > 2*Second half => depleted local rewards
		#	print("Depleted local rewards... Moving away now... First half:{} Second half:{}".format(rewards[:episode_length//2].sum(),rewards[episode_length//2:].sum()))
			self.up_shift()
		else:
			if rewards.mean()==0:
			#	print("Zero Rewards :: Probably in the middle of nowhere !")
				self.up_shift()
			else:
			#	print("Auto-Decay !")
				self.down_shift()








	def get_action_repeat(self):
		return self.current_action_repeat

	def get_replay_buffer(self):
		return self.current_replay_buffer

	def get_optimizer(self):
		return self.current_optimizer

	def get_network(self):
		return self.current_network

	def save(self):		#saving only the models for now
		for i,net in enumerate(self.list_of_networks):
			net.save("ensemble_model_{}.pth".format(i))

	def load(self, list_of_files=None):		#saving only the models for now
		if list_of_files is None:
			for i,net in enumerate(self.list_of_networks):
				net.load("ensemble_model_{}.pth".format(i))
		else:
			for i,net in enumerate(self.list_of_networks):
				net.load(list_of_files[i])

	def up_shift(self):
		if self.current!=len(self.list_of_networks)-1:
			self.current+=1
		else:
			print("WARNING: Ensemble MAX limit reached :: Cannot shift up", end='\r')

	def down_shift(self):
		if self.current!=0:
			self.current-=1
		"""
		Disabling prints cause of auto-decay
		else:
			print("WARNING: Ensemble MIN limit reached :: Cannot shift down", end='\r')
		"""

	def update_context(self):
		self.current_network = self.list_of_networks[self.current]
		self.current_optimizer = self.list_of_optimizers[self.current]
		self.current_replay_buffer = self.list_of_replay_buffers[self.current]
		self.current_action_repeat = self.list_of_action_repeats[self.current]
