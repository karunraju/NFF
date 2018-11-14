import sys, argparse
from Agent import Agent
from Agent_ppo import Agent_ppo

def parse_arguments():
  parser = argparse.ArgumentParser(description='Navigate-Fetch-Find Argument Parser')
  parser.add_argument('--render', dest='render', type=int, default=0)
  parser.add_argument('--train', dest='train', type=int, default=1)
  parser.add_argument('--model', dest='model_file', type=str, help='Model file')
  parser.add_argument('--method', dest='method', type=str, default='DoubleQ',
                      help='Duel or DoubleQ')
  return parser.parse_args()

def main():
  args = parse_arguments()

  #agent = Agent(render=args.render, method=args.method)
  agent = Agent_ppo(render=args.render)
  agent.train()
  #agent.test()

if __name__ == '__main__':
  main()
