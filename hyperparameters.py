BATCH_SIZE = 32

LEARNING_RATE = 0.001

DISCOUNT_FACTOR = 0.9

# Clone the Q network to target network for every C steps
C = 250

CLIP_VALUE = 5

TRAINING_TIME = 3000000

# REPLAY BUFFER
PRIORITIZED_REPLAY = True

REPLAY_MEMORY_SIZE = 2000

BURN_IN = 400

# A2C
A2C_EPISODE_SIZE_MIN = 50
A2C_EPISODE_SIZE_MAX = 100

N = 20

gamma = 1.0

A2C_SEQUENCE_LENGTH = 4

bidirectional = False

STATE_SIZE = 4

AUX_TASK_BATCH_SIZE = 32

INCEPTION_FILTER = True

SCENT_MODALITY = True
