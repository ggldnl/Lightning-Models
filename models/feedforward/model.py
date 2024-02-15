import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn, optim
import torchmetrics


class FeedForward(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.001,
    ):

        super(FeedForward, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate

        # Define the loss function
        self.loss_function = nn.CrossEntropyLoss()

        # Define metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=output_size)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def common_step(self, batch, batch_idx):

        x, y = batch['image'], batch['label']

        logits = self.forward(x)
        loss = self.loss_function(logits, y)

        target = y.argmax(dim=1)
        preds = logits.argmax(dim=1)

        return loss, preds, target

    def training_step(self, batch, batch_idx):

        loss, output, target = self.common_step(batch, batch_idx)

        # self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss, output, target = self.common_step(batch, batch_idx)

        accuracy = self.accuracy(output, target)
        f1_score = self.f1_score(output, target)

        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_f1', f1_score, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):

        loss, output, target = self.common_step(batch, batch_idx)

        accuracy = self.accuracy(output, target)
        f1_score = self.f1_score(output, target)

        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy, prog_bar=True)
        self.log('test_f1', f1_score, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
