from data import CustomMNISTDataModule
import pytorch_lightning as pl
from model import FeedForward
from config import *


if __name__ == "__main__":

    # Define the data
    datamodule = CustomMNISTDataModule(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Define the model
    model = FeedForward(
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE
    )

    # Define the trainer
    trainer = pl.Trainer(min_epochs=1, max_epochs=NUM_EPOCHS, precision=PRECISION)

    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)
