from models.transformer.blocks.multihead_attention import MultiHeadAttentionBlock
from models.transformer.blocks.positional_encoding import PositionalEncoding
from models.transformer.blocks.input_embeddings import InputEmbeddings
from models.transformer.blocks.feed_forward import FeedForwardBlock
from models.transformer.blocks.projection import ProjectionLayer
from models.transformer.blocks.encoder_block import EncoderBlock
from models.transformer.blocks.decoder_block import DecoderBlock
from models.transformer.blocks.encoder import Encoder
from models.transformer.blocks.decoder import Decoder
import pytorch_lightning as pl
import torch.nn as nn
import torch


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 source_embedding: InputEmbeddings,
                 target_embedding: InputEmbeddings,
                 source_position_encoding: PositionalEncoding,
                 target_position_encoding: PositionalEncoding,
                 projection_layer: ProjectionLayer
                 ) -> None:

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding  # Input embedding for the input sequence (language 1)
        self.target_embedding = target_embedding  # Output embedding for the output sequence (language 2)
        self.source_position_encoding = source_position_encoding
        self.target_position_encoding = target_position_encoding
        self.projection_layer = projection_layer

    def encode(self, source, source_mask):

        # Apply source embeddings to the input source language
        source = self.source_embedding(source)

        # Apply source positional encoding to the source embeddings
        source = self.source_position_encoding(source)

        # Return the source embeddings plus a source mask to prevent attention to certain elements
        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):

        # Apply target embeddings to the input target language
        target = self.target_embedding(target)

        # Apply target positional encoding to the target embeddings
        target = self.target_position_encoding(target)

        # The target mask ensures that the model won't see future elements of the sequence
        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = self.project(decoder_output)
        return proj_output

    def translate(self, input_sequence, source_tokenizer, target_tokenizer, max_output_len, verbose=True):

        # Build the encoder input
        encoder_input = source_tokenizer.get_encoder_input(input_sequence)
        encoder_input = encoder_input.unsqueeze(0)  # Add the batch dimension
        source_mask = source_tokenizer.get_encoder_mask(encoder_input)

        # Compute the output of the encoder for the source sequence
        encoder_output = self.encode(encoder_input, source_mask)

        # Initialize the decoder input with the SOS token
        decoder_input = torch.empty(1, 1).fill_(target_tokenizer.sos_token_id).type(torch.int64)

        while True:

            # If the output sequence length is past the max output length, break the loop
            if decoder_input.size(1) == max_output_len:
                break

            # Build a mask for the decoder input
            decoder_mask = target_tokenizer.get_decoder_mask(decoder_input)

            # Compute the output of the decoder
            out = self.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # Apply the projection layer to get the probabilities for the next token
            prob = self.project(out[:, -1])

            # Select token with the highest probability
            _, predicted_token_tensor = torch.max(prob, dim=1)
            predicted_token_id = predicted_token_tensor.item()
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).fill_(predicted_token_id).type(torch.int64)], dim=1)

            if verbose:
                print(f'{predicted_token_id} ', end='')

            # If the next token is an EOS token, break the loop
            if predicted_token_id == target_tokenizer.eos_token_id:
                break

        if verbose:
            print()

        # Sequence of tokens generated by the decoder excluding the SOS and including the EOS
        return decoder_input.squeeze(0).tolist()[1:]

    @classmethod
    def build(cls,
              src_vocab_size: int,
              tgt_vocab_size: int,
              src_seq_len: int,
              tgt_seq_len: int,
              embed_dim: int,
              num_encoders: int,
              num_decoders: int,
              heads: int,
              dropout: float,
              d_ff: int
              ):
        """
        Build a transformer module
        """

        # Create the embedding layers
        source_embeddings = InputEmbeddings(embed_dim, src_vocab_size)
        target_embeddings = InputEmbeddings(embed_dim, tgt_vocab_size)

        # Crete positional encoding layers
        source_position_encoding = PositionalEncoding(embed_dim, src_seq_len, dropout)
        target_position_encoding = PositionalEncoding(embed_dim, tgt_seq_len, dropout)

        # Create a list of encoder blocks
        encoder_blocks = []
        for _ in range(num_encoders):

            encoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)
            feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # Create a list of decoder blocks
        decoder_blocks = []
        for _ in range(num_decoders):
            decoder_self_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(embed_dim, heads, dropout)
            feed_forward_block = FeedForwardBlock(embed_dim, d_ff, dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                         feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)

        # Create the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = ProjectionLayer(embed_dim, tgt_vocab_size)

        # Create the transformer
        transformer = Transformer(
            encoder,
            decoder,
            source_embeddings,
            target_embeddings,
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


class LightningTransformer(pl.LightningModule):

    def __init__(self, model, criterion, optimizer, scheduler=None):
        super(LightningTransformer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def translate(self, input_sequence, source_tokenizer, target_tokenizer, max_output_len, verbose=True):
        return self.model.translate(input_sequence, source_tokenizer, target_tokenizer, max_output_len, verbose)

    def common_step(self, batch, batch_idx):

        encoder_input = batch['encoder_input']  # (batch, seq_len)
        decoder_input = batch['decoder_input']  # (batch, seq_len)
        encoder_mask = batch['encoder_mask']    # (batch, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask']    # (batch, 1, seq_len, seq_len)
        label = batch['label']                  # (batch, seq_len)

        # Run the input through the model
        output = self.model(encoder_input, encoder_mask, decoder_input, decoder_mask)
        loss = self.criterion(output.view(-1, output.size(-1)), label.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        if self.scheduler:
            return {'optimizer': optimizer, 'lr_scheduler': self.scheduler}
        else:
            return optimizer
