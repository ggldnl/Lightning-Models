import pytorch_lightning as pl
from layer_norm import LayerNorm
import torch.nn as nn


class ResidualConnection(pl.LightningModule):

    def __init__(self,
                 dropout: float
                 ):

        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        # We combine the x with the output of the next layer, then we apply the dropout.
        # This is the definition of add and norm
        return x + self.dropout(sublayer(self.norm(x)))
