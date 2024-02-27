import pytorch_lightning as pl
from models.transformer.blocks.encoder import Encoder
from models.transformer.blocks.decoder import Decoder
from models.transformer.blocks.encoder_block import EncoderBlock
from models.transformer.blocks.decoder_block import DecoderBlock
from models.transformer.blocks.input_embedding import InputEmbeddings
from models.transformer.blocks.positional_encoding import PositionalEncoding
from models.transformer.blocks.multihead_attention import MultiHeadAttention
from models.transformer.blocks.feedforward import FeedForward
from models.transformer.blocks.projection import Projection
from torch import nn, optim


class Transformer(pl.LightningModule):

    def __init__(self,
                 encoder,
                 decoder,
                 source_embedding,  # Input embedding for the input sequence (language 1)
                 target_embedding,  # Output embedding for the output sequence (language 2)
                 source_position_encoding,
                 target_position_encoding,
                 projection,
                 learning_rate,
                 loss_fn
                 ):

        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position_encoding = source_position_encoding
        self.target_position_encoding = target_position_encoding
        self.projection = projection
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

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

    """
    def generate_source_mask(self, source):
        # For each element, if the element is a source_pad then it will be set to 0
        return (source != self.source_pad).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, sequence_len)

    def generate_target_mask(self, target):
        return ((target != self.source_pad).unsqueeze(0).unsqueeze(0).int() &
                causal_mask(target.size(0)))
    
    def infer(self, source):

        # Generate source mask
        source_mask = self.generate_source_mask(source)

        # Encoder forward pass
        encoder_output = self.encode(source, source_mask)

        # Initialize target sequence for decoding (e.g., start with SOS token)
        target_sequence = torch.tensor([self.target_sos])

        # Generate target mask for the initial token
        target_mask = causal_mask(target_sequence.size(0))

        # Perform step-by-step decoding
        for _ in range(self.max_target_length):  # Define max_target_length based on your application
            # Decoder forward pass for the current token
            decoder_output = self.decode(encoder_output, source_mask, target_sequence, target_mask)

            # Project to output space
            output_token = self.project(decoder_output[-1:])  # Taking the last generated token

            # Append the generated token to the target sequence
            target_sequence = torch.cat([target_sequence, output_token.argmax(dim=-1)])

            # Update target mask for the newly generated token
            target_mask = causal_mask(target_sequence.size(0))

        return target_sequence[1:]  # Exclude the SOS token from the output
    """

    def common_step(self, batch, batch_idx):

        encoder_input = batch['encoder_input']  # (batch, seq_len)
        decoder_input = batch['decoder_input']  # (batch, seq_len)
        encoder_mask = batch['encoder_mask']  # (batch, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask']  # (batch, 1, seq_len, seq_len)
        label = batch['label']  # (batch, seq_len)

        # Run the input through the model
        encoder_output = self.encode(encoder_input, encoder_mask)  # (batch, seq_len, embedding_size)
        decoder_output = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (batch, seq_len, embedding_size)
        projection_output = self.project(decoder_output)  # (batch, seq_len, target_vocab_size)

        # Compare to the label and compute loss
        projection_output = projection_output.view(-1, self.target_embedding.vocab_size)
        label = label.view(-1).long()
        loss = self.loss_fn(projection_output, label)
        accuracy = 0

        return loss, accuracy

    def training_step(self, batch, batch_idx):

        loss, accuracy = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        loss, accuracy = self.common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    @classmethod
    def build(cls,
              source_vocab_size,
              target_vocab_size,
              source_sequence_length,
              target_sequence_length,
              learning_rate,
              loss_fn,
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
            projection_layer,
            learning_rate,
            loss_fn
        )

        # Initialize the model parameters to train faster (otherwise they are initialized with
        # random values). We use Xavier initialization
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer
