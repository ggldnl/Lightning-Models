from architecture.blocks.multihead_attention import MultiHeadAttentionBlock
from architecture.blocks.residual_connection import ResidualConnection
from architecture.blocks.feed_forward import FeedForwardBlock
import torch.nn as nn


class DecoderBlock(nn.Module):
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
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float
                 ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout) for _ in range(3)
        ])

    def forward(self,
                x,                  # Input to the decoder
                encoder_output,     # output of the encoder
                source_mask,        # Mask applied to the encoder
                target_mask         # Mask applied to the decoder
                ):

        # We need two masks since we are translating a sequence into another
        # and the output sequence can have a different set of  tokens we might
        # want to exclude
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))

        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 source_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
