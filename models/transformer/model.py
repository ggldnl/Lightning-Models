import pytorch_lightning as pl
import torch

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
                 loss_fn,
                 source_tokenizer=None,  # Used for basic logging
                 target_tokenizer=None  # Used for basic logging
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

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_position_encoding(source)
        return self.encoder(source, source_mask)

    def decode(self, source, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position_encoding(target)
        return self.decoder(target, source, source_mask, target_mask)

    def project(self, x):
        return self.projection(x)

    def translate(self,
                  input_text,
                  source_tokenizer,
                  target_tokenizer,
                  max_output_length=20,
                  ):
        """
        Input sequence contain token ids
        """

        source_sos_token_id = source_tokenizer.sos_token_id
        source_eos_token_id = source_tokenizer.eos_token_id
        source_pad_token_id = source_tokenizer.pad_token_id

        target_sos_token_id = target_tokenizer.sos_token_id
        target_eos_token_id = target_tokenizer.eos_token_id

        # Translate the sequence
        self.eval()
        with torch.no_grad():

            input_sequence = source_tokenizer.encode(input_text)
            enc_padding_tokens = self.source_position_encoding.seq_len - len(input_sequence) - 2

            # Unsqueeze adds the batch dimension
            encoder_input = torch.cat(
                [
                    torch.tensor([source_sos_token_id], dtype=torch.int32),
                    torch.tensor(input_sequence, dtype=torch.int32),
                    torch.tensor([source_eos_token_id], dtype=torch.int32),
                    torch.tensor([source_pad_token_id] * enc_padding_tokens, dtype=torch.int32)
                ]
            ).unsqueeze(0)
            encoder_mask = (encoder_input != source_pad_token_id).unsqueeze(0).unsqueeze(0).int()
            encoder_output = self.encode(encoder_input, encoder_mask)

            # Initialize the decoder input with the sos token
            decoder_input = torch.tensor([[target_sos_token_id]])

            # Generate the translation word by word
            while decoder_input.size(1) < max_output_length:

                # Build a mask for target and compute the output
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int32)
                out = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                # Project the decoder output to get probabilities over the target vocabulary
                prob = self.project(out[:, -1])
                _, predicted_token = torch.max(prob, dim=1)
                predicted_token_id = predicted_token.item()

                decoder_input = torch.cat(
                    [
                        decoder_input,
                        predicted_token.unsqueeze(0)
                    ], dim=1
                )

                print(f"{target_tokenizer.id_to_token(predicted_token_id)} ", end=' ')

                # Break if we predict the end of sentence token
                if predicted_token_id == target_eos_token_id:
                    break

        # return target_tokenizer.decode(decoder_input.tolist()[0][1:])
        return decoder_input.tolist()[0][1:]

    """
    def translate(self, input_text, source_tokenizer, target_tokenizer, max_output_length=50):

        source_sos_token_id = source_tokenizer.sos_token_id
        source_eos_token_id = source_tokenizer.eos_token_id
        source_pad_token_id = source_tokenizer.pad_token_id

        target_sos_token_id = target_tokenizer.sos_token_id
        target_eos_token_id = target_tokenizer.eos_token_id

        with torch.no_grad():

            input_sequence = source_tokenizer.encode(input_text)

            # Encode the input sequence and input mask to get encoder output
            encoder_input = torch.tensor([input_sequence])
            encoder_mask = torch.ones_like(encoder_input).unsqueeze(1).unsqueeze(1)
            encoder_output = self.encode(encoder_input, encoder_mask)

            # Initialize decoder input with SOS token and the decoder mask
            decoder_input = torch.tensor([[target_sos_token_id]])
            decoder_mask = torch.ones_like(decoder_input).unsqueeze(1).unsqueeze(2)

            generated_tokens = []

            for _ in range(max_output_length):

                # Decode one step at a time
                decoder_output = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                # Project the decoder output to get probabilities over the target vocabulary
                projection_output = self.project(decoder_output)

                # Get the token with the maximum probability
                predicted_token_id = projection_output[0][0].argmax().item()

                # Break if EOS token is predicted or max length is reached
                if predicted_token_id == target_eos_token_id or len(generated_tokens) >= max_output_length:
                    break

                # Append the predicted token to the generated sequence
                generated_tokens.append(predicted_token_id)

                # print(f'Predicted token: {predicted_token_id}')

                # Append the generated token to the decoder input for the next step
                predicted_token = torch.tensor([[predicted_token_id]])
                decoder_input = torch.cat([decoder_input, predicted_token])

                # Update the attention mask for the decoder
                decoder_mask = torch.ones_like(decoder_input).unsqueeze(1).unsqueeze(2)

            # Return the generated tokens excluding the SOS token
            return generated_tokens[:1]
    """

    def common_step(self, batch, batch_idx):

        encoder_input = batch['encoder_input']  # (batch, seq_len)
        decoder_input = batch['decoder_input']  # (batch, seq_len)
        encoder_mask = batch['encoder_mask']  # (batch, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask']  # (batch, 1, seq_len, seq_len)
        label = batch['label']  # (batch, seq_len)

        # Run the input through the model
        encoder_output = self.encode(encoder_input, encoder_mask)  # (batch, seq_len, embedding_size)
        decoder_output = self.decode(encoder_output, encoder_mask, decoder_input,
                                     decoder_mask)  # (batch, seq_len, embedding_size)
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
              source_tokenizer=None,
              target_tokenizer=None
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
            loss_fn,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer
        )

        # Initialize the model parameters to train faster (otherwise they are initialized with
        # random values). We use Xavier initialization
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer
