from models.feedforward.MNIST_classification.data import CustomMNISTDataModule
import pytorch_lightning as pl
from models.feedforward.model import FeedForward
from models.feedforward.MNIST_classification.config import *


if __name__ == "__main__":

    # Define the data
    datamodule = CustomMNISTDataModule(
        data_dir=DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Define the model
    model = FeedForward(
        INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE
    )

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')

    # Define the trainer
    trainer = pl.Trainer(min_epochs=1, max_epochs=NUM_EPOCHS, precision=PRECISION, callbacks=[checkpoint_callback])

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
