import numpy as np
class ReplayMemory():

  def __init__(self, memory_size=50000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from
    # the randomly initialized agent. Memory size is the maximum size after which old
    # elements in the memory are replaced.
    self.memory_size = memory_size
    self.burn_in = burn_in

    # List to store transition tuples
    self.mem = [()]*self.memory_size
    self.idx = 0

    # Flag to check if the entire buffer is filled for the first time
    self.full = False

  def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action,
    # reward, next state, terminal flag tuples. You will feed this to your model to train.
    if self.full:
      ids = np.random.randint(0, self.memory_size, 32)
    else:
      ids = np.random.randint(0, self.idx, 32)

    return [self.mem[i] for i in ids]

  def append(self, transition):
    # Appends transition to the memory.
    self.mem[self.idx] = transition
    self.idx = self.idx + 1
    if self.idx >= self.memory_size:
      self.idx = 0
      self.full = True

  def get_burn_in(self):
    return self.burn_in
