from transformer_block import TransformerBlock
import pytorch_lightning as pl
import torch.nn as nn
import torch


class Encoder(pl.LightningModule):
    """
                                                      |
                                  +--------------------------------------+
                                  |                   |                  |
                                  |                  ...                 |
                                  |                                      |
                             Nx   |              Transformer             |
                                  |                 Block                |
                                  |                                      |
                                  |                  ...                 |
                                  |                   |                  |
                                  +--------------------------------------+
     +------------------------+                       ^
     |                        |                       |
     |  Positional Encoding   | --------------------- +
     |                        |                       ^
     +------------------------+                       |
                                  +--------------------------------------+
                                  |                                      |
                                  |             Input Embedding          |
                                  |                                      |
                                  +--------------------------------------+
                                                      ^
                                                      |
                                                    Inputs

    """

    def __init__(self,
                 source_vocab_size,  # Needed for embedding computation
                 max_length,  # How long is the max sequence length, needed for positional embedding computation
                 num_layers,  # Number of Transformer Blocks

                 # Transformer block parameters
                 embed_size,
                 heads,
                 dropout,
                 forward_expansion
                 ):

        super(Encoder, self).__init__()

        self.embed_size = embed_size

        # torch.nn.Embedding is a simple lookup table that stores embeddings of a fixed dictionary size.
        # We can then retrieve them using indices
        self.token_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                ) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        # N samples of seq_length length
        N, seq_length = x.shape

        # Create the positional embeddings. We have a tensor of N times a (0, seq_lengt) range
        positions = torch.arange(0, seq_length).expand(N, seq_length)

        out = self.dropout(self.token_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # In the encoder all the inputs are going to be the same
            out = layer(out, out, out, mask)

        return out