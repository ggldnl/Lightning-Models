
EMBED_DIM = 100
NUM_ENCODERS = 2  # Number of encoder blocks
NUM_DECODERS = 6  # Number of decoder blocks
DROPOUT = 0.1
HEADS = 4
D_FF = 500

BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 350
PRECISION = '16-mixed'

LOGS_DIR = 'logs'
DATA_DIR = 'dataset'
TOK_DIR = 'tokenizers'
