import sys
from canonical_plot import plot

def main(filename):
  with open(filename, 'r') as f:
    rewards = f.read().splitlines()

  rewards = [float(r) for r in rewards]
  rew = [0]*len(rewards)*100
  for i, r in enumerate(rewards):
   rew[(i+1)*100 - 1] =  r  * 100

  plot('./', rew)

if __name__ == '__main__':
  main(sys.argv[1])
