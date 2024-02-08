import pytorch_lightning as pl
from encoder import Encoder
from decoder import Decoder
import torch


class Transformer(pl.LightningModule):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 source_pad_idx,
                 target_pad_idx,
                 embed_size,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 max_length=100
                 ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            source_vocab_size,
            max_length,
            num_layers,
            embed_size,
            heads,
            dropout,
            forward_expansion
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx

    def make_source_mask(self, source):

        # source_mask shape: (N, 1, 1, len(source))
        return (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_target_mask(self, target):

        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(N, 1, target_len, target_len)
        return target_mask

    def forward(self, source, target):

        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)

        source_encoder = self.encoder(source, source_mask)

        out = self.decoder(target, source_encoder, source_mask, target_mask)

        return out
