import sys, argparse
from Agent import Agent
from Agent_aux import Agent_aux

def parse_arguments():
  parser = argparse.ArgumentParser(description='Navigate-Fetch-Find Argument Parser')
  parser.add_argument('--render', dest='render', type=int, default=0)
  parser.add_argument('--train', dest='train', type=int, default=1)
  parser.add_argument('--model', dest='model_file', type=str, default=None,
                      help='Model file')
  parser.add_argument('--random', dest='random', type=bool, default=False,
                      help='Runs Random Method')
  parser.add_argument('--method', dest='method', type=str, default='DoubleQ',
                      help='Duel or DoubleQ')
  return parser.parse_args()

def main():
  args = parse_arguments()

  #agent = Agent(render=args.render, method=args.method)
  agent = Agent_aux(render=args.render)
  if args.random:
    agent.run_random_policy()
  elif not args.train:
    if args.model_file is None:
      raise ValueError('Require model file')
    agent.test(model_file=args.model_file)
  else:
    agent.train()

if __name__ == '__main__':
  main()
