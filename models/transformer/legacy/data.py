from torch.utils.data import Dataset
import torch.nn as nn
import torch


class BilingualDataset(Dataset):

    def __init__(self,
                 dataset,
                 source_tokenizer,
                 target_tokenizer,
                 source_language,  # Name of the source language
                 target_language,  # Name of the target language
                 sequence_len
                 ):
        super(BilingualDataset, self).__init__()

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_len = sequence_len

        # Special tokens
        # self.sos_token = torch.Tensor([source_tokenizer.token_to_id(['[SOS]'])], dtype=torch.int64)
        # self.eos_token = torch.Tensor([source_tokenizer.token_to_id(['[EOS]'])], dtype=torch.int64)
        # self.pad_token = torch.Tensor([source_tokenizer.token_to_id(['[PAD]'])], dtype=torch.int64)

        # Get special token IDs
        sos_id = source_tokenizer.token_to_id('[SOS]')
        eos_id = source_tokenizer.token_to_id('[EOS]')
        pad_id = source_tokenizer.token_to_id('[PAD]')

        # Convert to PyTorch tensors
        self.sos_token = torch.tensor([sos_id], dtype=torch.long)
        self.eos_token = torch.tensor([eos_id], dtype=torch.long)
        self.pad_token = torch.tensor([pad_id], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        source_target_pair = self.dataset[item]

        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        # This gives the IDs of each word in the original sentence
        encoder_input_tokens = self.source_tokenizer.encode(source_text).ids
        decoder_input_tokens = self.target_tokenizer.encode(target_text).ids

        encoder_num_padding_tokens = self.sequence_len - len(encoder_input_tokens) - 2  # Minus EOS and SOS
        decoder_num_padding_tokens = self.sequence_len - len(decoder_input_tokens) - 1  # We only have SOS

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Build the tensors. One sentence will be sent to the input of the encoder,
        # one sentence will be sent to the input of the decoder and one sentence will
        # be the expected output of the decoder (label)

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Build the decoder input. We only add SOS
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Build the label. We only add EOS (what we expect as the output from the decoder)
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.sequence_len
        assert decoder_input.size(0) == self.sequence_len
        assert label.size(0) == self.sequence_len

        return {
            "encoder_input": encoder_input,  # (sequence_len)
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, sequence_len)

            # Words can only look at words coming before them
            # (1, sequence_len) & (1, sequence_len, sequence_len) (broadcasting)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() &
                            causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text": source_text,
            "target_text": target_text
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
