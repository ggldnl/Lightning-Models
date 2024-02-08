# Model related hyperparameters
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = INPUT_SIZE // 10
OUTPUT_SIZE = 10

# Train related hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001

# Dataset related hyperparameters
DATA_DIR = "dataset/"
NUM_WORKERS = 4

# Computer related hyperparameters
# ACCELERATOR = "gpu"
# DEVICES = 2
PRECISION = 16
