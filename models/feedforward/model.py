import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim
import torchmetrics


class FeedForward(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_fn=F.relu,
        loss_fn=nn.CrossEntropyLoss,
        learning_rate=0.001,
    ):

        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn()
        self.learning_rate = learning_rate

        # Metrics
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=output_size
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=output_size)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
