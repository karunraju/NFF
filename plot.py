import sys
from canonical_plot import plot

def main(filename):
  with open(filename, 'r') as f:
    rewards = f.read().splitlines()

  rewards = [float(r) for r in rewards]
  a = [0]*99
  rew = []
  for r in rewards:
   rew = rew + [r*100] + a

  plot('./', rew)

if __name__ == '__main__':
  main(sys.argv[1])
