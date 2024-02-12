import pytorch_lightning as pl
import torch
import torch.nn as nn


class FeedForward(pl.LightningModule):
    """
    Fully connected layer that the model uses both in the encoder and in the decoder.
    In the paper, it is described as:

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    max(0, x) == ReLU(x)
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float
                 ):

        super(FeedForward, self).__init__()

        # By default, Linear has bias=True that automatically creates a bias for us
        self.linear_1 = nn.Linear(d_model, d_ff)  # W_1 and b_1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W_2 and b_2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
