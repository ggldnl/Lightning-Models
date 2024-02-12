import pytorch_lightning as pl
from blocks.encoder import Encoder
from blocks.decoder import Decoder
from blocks.encoder_block import EncoderBlock
from blocks.decoder_block import DecoderBlock
from blocks.input_embedding import InputEmbeddings
from blocks.positional_encoding import PositionalEncoding
from blocks.multihead_attention import MultiHeadAttention
from blocks.feedforward import FeedForward
from blocks.projection import Projection
import torch.nn as nn
import torch


class Transformer(pl.LightningModule):

    def __init__(self,
                 encoder,
                 decoder,
                 source_embedding,  # Input embedding for the input sequence (language 1)
                 target_embedding,  # Output embedding for the output sequence (language 2)
                 source_position_encoding,
                 target_position_encoding,
                 projection
                 ):

        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position_encoding = source_position_encoding
        self.target_position_encoding = target_position_encoding
        self.projection = projection

    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_position_encoding(source)
        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position_encoding(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self.projection(x)

    @classmethod
    def build(cls,
              source_vocab_size,
              target_vocab_size,
              source_sequence_length,
              target_sequence_length,
              embedding_size: int = 512,
              num_encoders: int = 6,  # Number of encoder blocks
              num_decoders: int = 6,  # Number of decoder blocks
              dropout: float = 0.1,
              heads: int = 8,
              d_ff: int = 2048,
              ):

        # Create embedding layers
        source_embedding = InputEmbeddings(embedding_size, source_vocab_size)
        target_embedding = InputEmbeddings(embedding_size, target_vocab_size)

        # Create the positional encoding layers
        source_position_encoding = PositionalEncoding(embedding_size, source_sequence_length, dropout)
        target_position_encoding = PositionalEncoding(embedding_size, target_sequence_length, dropout)

        # Create encoder blocks
        encoder_blocks = []
        for _ in range(num_encoders):
            encoder_self_attention_block = MultiHeadAttention(embedding_size, heads, dropout)
            feed_forward_block = FeedForward(embedding_size, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # Create encoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))

        # Create decoder blocks
        decoder_blocks = []
        for _ in range(num_decoders):
            decoder_self_attention_block = MultiHeadAttention(embedding_size, heads, dropout)
            decoder_cross_attention_block = MultiHeadAttention(embedding_size, heads, dropout)
            decoder_feed_forward_block = FeedForward(embedding_size, d_ff, dropout)
            decoder_block = DecoderBlock(
                decoder_self_attention_block,
                decoder_cross_attention_block,
                decoder_feed_forward_block,
                dropout
            )
            decoder_blocks.append(decoder_block)

        # Create decoder
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = Projection(embedding_size, target_vocab_size)

        transformer = Transformer(
            encoder,
            decoder,
            source_embedding,
            target_embedding,
            source_position_encoding,
            target_position_encoding,
            projection_layer
        )

        # Initialize the model parameters to train faster (otherwise they are initialized with
        # random values). We use Xavier initialization
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer


if __name__ == '__main__':

    # We have a batch of two sample inputs and the two respective targets

    # 1 is for SOS
    # 0 is for PAD
    # 2 is for EOS
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])

    source_vocab_size = 10
    target_vocab_size = 10
    source_sequence_length = 10
    target_sequence_length = 10

    model = Transformer.build(
        source_vocab_size,
        target_vocab_size,
        source_sequence_length,
        target_sequence_length,
        embedding_size=10,
        heads=2
    )

    out = model(x, trg[:, :-1])
    print(out)
