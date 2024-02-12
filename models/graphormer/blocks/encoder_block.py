import pytorch_lightning as pl
import torch.nn as nn

from models.transformer.blocks.layer_norm import LayerNorm


class EncoderBlock(pl.LightningModule):
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
    #    |        +--------------------------------------+   |
    #    |        |                                      |   |
    #    |  +---->|             Add & Norm               |   |
    #    |  |     |                                      |   |
    #    |  |     +--------------------------------------+   |
    #    |  |                       |                        |
    #    |  |    +---------------------------------------+   |
    #    |  |    |                                       |   |
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
                 feed_forward_block,
                 dropout
                 ):

        super(EncoderBlock, self).__init__()

        self.self_attention = self_attention_block
        self.feed_forward = feed_forward_block
        self.norm_1 = LayerNorm()
        self.norm_2 = LayerNorm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, source_mask, spatial_encoding, edge_encoding, centrality):

        # We might want to use a source_mask on the input of the encoder
        # to exclude some tokens, such as the <EOS> token and so on...
        # We don't want these tokens to interact with other tokens

        # Input goes to the Multi-Head Self Attention module. It's the
        # sequence that is watching itself. Each token of the sequence
        # is interacting with other tokens of the same sequence.
        # In the decoder we have a different situation (Cross Attention)
        # where the query coming from the decoder is watching the keys
        # and the values coming from the encoder
        attention_out = self.self_attention(x, x, x, source_mask, spatial_encoding, edge_encoding, centrality)

        # Output of the Multi-Head Attention goes to Add & Norm.
        # We sum query for the skip connection
        attention_out = self.dropout(self.norm_1(x + attention_out))

        feedforward_out = self.feed_forward(attention_out)

        feedforward_out = self.dropout(self.norm_2(feedforward_out + attention_out))

        return feedforward_out
