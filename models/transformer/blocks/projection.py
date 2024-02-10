import pytorch_lightning as pl
import torch.nn as nn
import torch


class Projection(pl.LightningModule):

    def __init__(self,
                 d_model,
                 vocab_size
                 ):
        """
        We expect the output of the decoder to have shape (batch, seq_len, d_model).
        We want to map this output back to the vocabulary by converting the embedding
        coming from the decoder to a position on it.
        """

        super(Projection, self).__init__()

        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)
