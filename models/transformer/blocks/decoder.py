import pytorch_lightning as pl
import torch.nn as nn
import torch
from decoder_block import DecoderBlock


class Decoder(pl.LightningModule):

    def __init__(self,
                 target_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 max_len
                 ):

        super(Decoder, self).__init__()

        self.token_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout
                ) for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, source_mask, target_mask):

        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length)

        x = self.droput(self.token_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # values and keys for the transformer block inside the decoder block come from the encoder
            x = layer(x, encoder_out, encoder_out, source_mask, target_mask)

        out = self.fc_out(x)

        return out

