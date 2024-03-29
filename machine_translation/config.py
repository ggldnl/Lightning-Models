EMBED_DIM = 512
NUM_ENCODERS = 2  # Number of encoder blocks
NUM_DECODERS = 6  # Number of decoder blocks
DROPOUT = 0.1
HEADS = 8
D_FF = 1024

BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 350
PRECISION = '16-mixed'

DATA_DIR = 'dataset'
TOK_DIR = 'tokenizers'
