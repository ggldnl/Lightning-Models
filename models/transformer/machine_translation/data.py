from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import numpy as np
import config
import torch
import utils
import os


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class OPUSDataset(Dataset):

    def __init__(self, data, max_seq_len, source_tokenizer=None, target_tokenizer=None):
        """
        The transformer is a sequence-to-sequence model used for translation
        from a language to another. The two languages might use a different
        set of tokens.
        """

        super(OPUSDataset, self).__init__()

        self.data = data
        self.max_seq_len = max_seq_len
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # Take the special tokens from the tokenizers (just to simplify the code)
        if source_tokenizer is not None:
            self.source_pad_token = self.source_tokenizer.token_to_id(self.source_tokenizer.pad_token)
            self.source_sos_token = self.source_tokenizer.token_to_id(self.source_tokenizer.sos_token)
            self.source_eos_token = self.source_tokenizer.token_to_id(self.source_tokenizer.eos_token)

        if target_tokenizer is not None:
            self.target_pad_token = self.target_tokenizer.token_to_id(self.target_tokenizer.pad_token)
            self.target_sos_token = self.target_tokenizer.token_to_id(self.target_tokenizer.sos_token)
            self.target_eos_token = self.target_tokenizer.token_to_id(self.target_tokenizer.eos_token)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        source_sentence, target_sentence = self.data[item]

        if self.source_tokenizer is None or self.target_tokenizer is None:
            return {
                "source_text": source_sentence,
                "target_text": target_sentence
            }

        source_token_ids = self.source_tokenizer.encode(source_sentence)
        target_token_ids = self.target_tokenizer.encode(target_sentence)

        enc_padding_tokens = self.max_seq_len - len(source_token_ids) - 2
        dec_padding_tokens = self.max_seq_len - len(target_token_ids) - 1

        assert enc_padding_tokens > 0, (f"Encoder input sentence is too long [{enc_padding_tokens}]. Try increasing "
                                        f"the maximum sequence length. Sentence: {source_sentence}.")

        assert dec_padding_tokens > 0, (f"Decoder input sentence is too long [{dec_padding_tokens}]. Try increasing "
                                        f"the maximum sequence length. Sentence: {target_sentence}.")

        # Add SOS, EOS and PAD tokens to the source text
        encoder_input = torch.cat(
            [
                torch.tensor([self.source_sos_token], dtype=torch.int32),
                torch.tensor(source_token_ids, dtype=torch.int32),
                torch.tensor([self.source_eos_token], dtype=torch.int32),
                torch.tensor([self.source_pad_token] * enc_padding_tokens, dtype=torch.int32)
            ]
        )

        # Add SOS and PAD tokens to the target text
        decoder_input = torch.cat(
            [
                torch.tensor([self.target_sos_token], dtype=torch.int32),
                torch.tensor(target_token_ids, dtype=torch.int32),
                torch.tensor([self.target_pad_token] * dec_padding_tokens, dtype=torch.int32)
            ]
        )

        # Add EOS and PAD tokens to the target text
        label = torch.cat(
            [
                torch.tensor(target_token_ids, dtype=torch.int32),
                torch.tensor([self.target_eos_token], dtype=torch.int32),
                torch.tensor([self.target_pad_token] * dec_padding_tokens, dtype=torch.int32)
            ]
        )

        assert encoder_input.size(0) == self.max_seq_len
        assert decoder_input.size(0) == self.max_seq_len
        assert label.size(0) == self.max_seq_len

        return {
            "encoder_input": encoder_input,  # (sequence_len)
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.source_pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1, 1, sequence_len)

            # Words can only look at words coming before them
            # (1, sequence_len) & (1, sequence_len, sequence_len) (broadcasting)
            "decoder_mask": (decoder_input != self.target_pad_token).unsqueeze(0).unsqueeze(0).int() &
                            causal_mask(decoder_input.size(0)),
            "label": label,
            "source_text": source_sentence,
            "target_text": target_sentence
        }


class OPUSDataModule(LightningDataModule):

    def __init__(self,
                 data_dir,
                 max_seq_len,
                 source_tokenizer=None,
                 target_tokenizer=None,
                 download=False,
                 batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS
                 ):
        super(OPUSDataModule, self).__init__()

        self.data_dir = data_dir
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_seq_len = max_seq_len
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers

        """
        Dataset link: 
        https://opus.nlpl.eu/ELRC-837-Legal/it&en/v1/ELRC-837-Legal#download
        """
        self.resource_url = "https://object.pouta.csc.fi/OPUS-Books/v1/tmx/en-it.tmx.gz"
        self.gz_path = os.path.join(self.data_dir, 'data.gz')
        self.tmx_path = os.path.join(self.data_dir, 'data.tmx')

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def prepare_data(self):

        if self.download:

            utils.download_resource(self.resource_url, self.gz_path)
            utils.extract_gz(self.gz_path, self.tmx_path)

    @staticmethod
    def parse_tmx(file_path):
        """
        Parse a TMX (Translation Memory eXchange) file and create a numpy 2D array.

        Parameters:
        - file_path (str): The path to the TMX file.

        Returns:
        - np.array: numpy array with two columns containing English
                        and Italian sentences respectively.
        """
        # Parse the XML file using ElementTree
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Define namespace mapping
        ns = {'xml': 'http://www.w3.org/XML/1998/namespace'}

        # Initialize empty list to store data
        data = []

        # Iterate through translation units (tu) in the TMX file
        for tu_element in root.findall('.//tu'):

            # Attempt to extract English and Italian segments
            en_tuv = tu_element.find('.//tuv[@xml:lang="en"]/seg', namespaces=ns)
            it_tuv = tu_element.find('.//tuv[@xml:lang="it"]/seg', namespaces=ns)

            # Check if either tuv element is not found
            if en_tuv is not None and it_tuv is not None:
                # Extract text and strip whitespace
                en_segment = en_tuv.text.strip()
                it_segment = it_tuv.text.strip()

                # Append segments to the data list
                data.append((en_segment, it_segment))

        return np.array(data)

    @staticmethod
    def train_text_val_split(data, train_percent, val_percent):

        # Calculate the sizes of each set
        num_samples = len(data)
        num_train = int(train_percent * num_samples)
        num_val = int(val_percent * num_samples)

        # Shuffle the indices to randomly select samples for each set
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split the indices into training, validation, and test sets
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Create DataFrames for each set using the selected indices
        train_data = data[train_indices]
        val_data = data[val_indices]
        test_data = data[test_indices]

        return train_data, test_data, val_data

    def setup(self, stage=None):

        # Put the content of the tmx into a dataframe
        data = self.parse_tmx(self.tmx_path)

        train_data, test_data, val_data = self.train_text_val_split(data, 0.7, 0.1)

        self.train_dataset = OPUSDataset(
            train_data,
            self.max_seq_len,
            self.source_tokenizer,
            self.target_tokenizer
        )

        self.test_dataset = OPUSDataset(
            test_data,
            self.max_seq_len,
            self.source_tokenizer,
            self.target_tokenizer
        )

        self.val_dataset = OPUSDataset(
            val_data,
            self.max_seq_len,
            self.source_tokenizer,
            self.target_tokenizer
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == '__main__':

    datamodule = OPUSDataModule(
        config.DATA_DIR,
        config.MAX_SEQ_LEN,
        download=False
    )

    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        source_text = batch['source_text']
        target_text = batch['target_text']
        print(f'Batch size: {len(source_text)}')
        print(f'Source:\n{source_text}')
        print(f'Target:\n{target_text}')
        break
