import pytorch_lightning as pl
import torch.nn as nn

from models.transformer.blocks.layer_norm import LayerNorm


class DecoderBlock(pl.LightningModule):
    #                               ^
    #                               |
    #    +---------------------------------------------------+
    #    |                          |                        |
    #    |        +--------------------------------------+   |
    #    |        |                                      |   |
    #    |  +---->|             Add & Norm               |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       |                        |
    #    |  |     +--------------------------------------+   |
    #    |  |     |                                      |   |
    #    |  |     |            Feed Forward              |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       ^                        |
    #    |  |                       |                        |
    #    |  +-----------------------+                        |
    #    |                          |                        |
    #    |                          |                        |
    #    |   +--------------------------------------+        |
    #    |   |                                      |        |
    #    |   |             Add & Norm               |<---+   |
    #    |   |                                      |    |   |
    #    |   +--------------------------------------+    |   |
    #    |                          |                    |   |
    #    |   +---------------------------------------+   |   |
    #    |   |                                       |   |   |
    #    |   |         Multi-Head Attention          |   |   |
    #    |   |                                       |   |   |
    #    |   +---------------------------------------+   |   |
    #    |            ^      ^      ^                    |   |
    #    |            |      |      |                    |   |
    #  --|------------+      |      |                    |   |
    #  --|-------------------+      +--------------------+   |
    #    |                          |                        |
    #    |        +--------------------------------------+   |
    #    |        |                                      |   |
    #    |  +---->|             Add & Norm               |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       |                        |
    #    |  |    +---------------------------------------+   |
    #    |  |    |               Masked                  |   |
    #    |  |    |         Multi-Head Attention          |   |
    #    |  |    |                                       |   |
    #    |  |    +---------------------------------------+   |
    #    |  |              ^        ^        ^               |
    #    |  |              |        |        |               |
    #    |  |              +--------+--------+               |
    #    |  |                       |                        |
    #    |  +-----------------------+                        |
    #    |                          |                        |
    #    +---------------------------------------------------+
    #                               |

    def __init__(self,
                 self_attention_block,
                 cross_attention_block,
                 feed_forward_block,
                 dropout
                 ):

        super(DecoderBlock, self).__init__()

        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.norm_1 = LayerNorm()
        self.norm_2 = LayerNorm()
        self.norm_3 = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,  # Input to the decoder
                encoder_output,  # Output of the encoder
                source_mask,  # Mask applied to the encoder
                target_mask  # Mask applied to the decoder
                ):

        # We need two masks since we are translating a sequence to another
        # and the output sequence could have tokens that we might want to
        # exclude that are different to the tokens we want to exclude in the
        # input sequence

        self_attention_out = self.self_attention(x, x, x, target_mask)

        # Output of the Multi-Head Attention goes to Add & Norm.
        # We sum query for the skip connection
        self_attention_out = self.dropout(self.norm_1(x + self_attention_out))

        # We give the cross attention block:
        # - the query coming from the decoder
        # - the keys and the values coming from the encoder
        # - the encoder mask
        cross_attention_out = self.cross_attention(self_attention_out, encoder_output, encoder_output, source_mask)
        cross_attention_out = self.dropout(self.norm_2(cross_attention_out + self_attention_out))

        feedforward_out = self.feed_forward(cross_attention_out)
        feedforward_out = self.dropout(self.norm_3(feedforward_out + cross_attention_out))

        return feedforward_out
