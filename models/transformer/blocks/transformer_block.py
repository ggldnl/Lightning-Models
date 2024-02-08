import pytorch_lightning as pl
from self_attention import SelfAttention
import torch.nn as nn


class TransformerBlock(pl.LightningModule):
    """
    Transformer block

                               ^
                               |
    +---------------------------------------------------+
    |                          |                        |
    |        +--------------------------------------+   |
    |        |                                      |   |
    |  +---->|             Add & Norm               |   |
    |  |     |                                      |   |
    |  |     +--------------------------------------+   |
    |  |                       |                        |
    |  |     +--------------------------------------+   |
    |  |     |                                      |   |
    |  |     |            Feed Forward              |   |
    |  |     |                                      |   |
    |  |     +--------------------------------------+   |
    |  |                       ^                        |
    |  |                       |                        |
    |  +-----------------------+                        |
    |                          |                        |
    |                          |                        |
    |        +--------------------------------------+   |
    |        |                                      |   |
    |  +---->|             Add & Norm               |   |
    |  |     |                                      |   |
    |  |     +--------------------------------------+   |
    |  |                       |                        |
    |  |    +---------------------------------------+   |
    |  |    |                                       |   |
    |  |    |         Multi-Head Attention          |   |
    |  |    |                                       |   |
    |  |    +---------------------------------------+   |
    |  |              ^        ^        ^               |
    |  |              |        |        |               |
    |  |              +--------+--------+               |
    |  |                       |                        |
    |  +-----------------------+                        |
    |                          |                        |
    +---------------------------------------------------+
                               |
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):

        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)

        # BatchNorm takes the average across the batch and then normalizes,
        # LayerNorm takes the average across every sample and then normalizes
        self.norm_1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.norm_2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        # Input goes to the Multi-Head Attention module
        attention = self.attention(value, key, query, mask)

        # Output of the Multi-Head Attention goes to Add & Norm.
        # We sum query for the skip connection
        x = self.dropout(self.norm_1(attention + query))

        forward = self.feed_forward(x)

        out = self.dropout(self.norm_2(forward + x))

        return out

