BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.9

# Clone the Q network to target network for every C steps
C = 250
CLIP_VALUE = 5
GRAD_CLIP_VAL = 5.0
TRAINING_TIME = 3000000# REPLAY BUFFER
PRIORITIZED_REPLAY = True
REPLAY_MEMORY_SIZE = 2000
BURN_IN = 400

# A2C
A2C_EPISODE_SIZE_MIN = 50
A2C_EPISODE_SIZE_MAX = 100
N = 50
gamma = 1.0
A2C_SEQUENCE_LENGTH = 10



AUX_TASK_BATCH_SIZE = 32
VFR_LOSS_WEIGHT = 1
RP_LOSS_WEIGHT = 1
PC_LOSS_WEIGHT = 0.1
REWARD_SHAPING = False
# HIND SIGHT EXPERIENCE REPLAY
HER = True
HER_DECAY = 0.9



STATE_SIZE = 4
ENSEMBLE=3
ACTION_REPEAT= [2,4,6]		#If ensemble is non-zero, actions repeat is expected to be a list
SWITCH_FREQUENCY = 20
INCEPTION_FILTER = True
SCENT_MODALITY = True
MLP_ACROSS_TIME = False
bidirectional = True

AGENTS=2