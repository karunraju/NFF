from Agent_aux import Agent_aux
from multiprocessing import Manager,Pool
import hyperparameters as PARAM
import sys, argparse
from aux.AuxNetwork import AuxNetwork
import torch.cuda as cuda

def parse_arguments():
	parser = argparse.ArgumentParser(description='Navigate-Fetch-Find Argument Parser')
	parser.add_argument('--render', dest='render', type=int, default=0)
	parser.add_argument('--train', dest='train', type=int, default=1)
	parser.add_argument('--model', dest='model_file', type=str, help='Model file')
	parser.add_argument('--method', dest='method', type=str, default='DoubleQ',
						help='Duel or DoubleQ')
	return parser.parse_args()	

def train(network):
	agent = Agent_aux(0,network)
	agent.train()

def main():
	man =Manager()
	if cuda.is_available():
		list_of_networks = man.list([AuxNetwork(state_size=PARAM.STATE_SIZE, action_space=3, seq_len=PARAM.A2C_SEQUENCE_LENGTH).cuda() for i in range(PARAM.ENSEMBLE)])		
	else:
		list_of_networks = man.list([AuxNetwork(state_size=PARAM.STATE_SIZE, action_space=3, seq_len=PARAM.A2C_SEQUENCE_LENGTH) for i in range(PARAM.ENSEMBLE)])		
	args = parse_arguments()
	p=Pool(PARAM.AGENTS)
	p.map(train, [list_of_networks]*PARAM.AGENTS, chunksize=1)


if __name__ == '__main__':
	main()
