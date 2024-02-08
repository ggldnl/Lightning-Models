import pytorch_lightning as pl
import torch.nn as nn
from self_attention import SelfAttention
from transformer_block import TransformerBlock


class DecoderBlock(pl.LightningModule):
    """
    Decoder block

                               ^
                               |
    +---------------------------------------------------+
    |                          |                        |
    |   +---------------------------------------+       |
    |   |                                       |       |
    |   |               Transformer             |<---+  |
    |   |                  Block                |    |  |
    |   |                                       |    |  |
    |   +---------------------------------------+    |  |
    |        ^        ^        ^                     |  |
    |        |        |        |                     |  |
   -|--------+--------+        |                     |  |
    |                          + --------------------+  |
    |                          |                        |
    |        +--------------------------------------+   |
    |        |                                      |   |
    |  +---->|             Add & Norm               |   |
    |  |     |                                      |   |
    |  |     +--------------------------------------+   |
    |  |                       |                        |
    |  |    +---------------------------------------+   |
    |  |    |               Masked                  |   |
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

    def __init__(self,
                 embed_size,
                 heads,
                 forward_expansion,
                 dropout
                 ):

        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)

        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):

        attention = self.attention(x, x, x, target_mask)

        # Skip connection
        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(value, key, query, source_mask)

        return out
