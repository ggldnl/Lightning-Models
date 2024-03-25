from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import pytorch_lightning as pl
import numpy as np
import config
import utils
import os


class OPUSDataset(Dataset):

    def __init__(self,
                 data,
                 max_seq_len,
                 source_tokenizer=None,
                 target_tokenizer=None,
                 ):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        source_sentence, target_sentence = self.data[item]

        if self.source_tokenizer is None or self.target_tokenizer is None:
            return {
                "source_text": source_sentence,
                "target_text": target_sentence
            }

        # Tokenize the sequences and convert them to ids
        encoder_input = self.source_tokenizer.get_encoder_input(source_sentence, self.max_seq_len)
        decoder_input = self.target_tokenizer.get_decoder_input(target_sentence, self.max_seq_len)
        encoder_mask = self.source_tokenizer.get_encoder_mask(encoder_input)
        decoder_mask = self.target_tokenizer.get_decoder_mask(decoder_input)
        label = self.target_tokenizer.get_label(target_sentence, self.max_seq_len)

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': encoder_mask,
            'decoder_mask': decoder_mask,
            'label': label,
            'source_text': source_sentence,
            'target_text': target_sentence
        }


class OPUSDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 max_seq_len,
                 source_tokenizer=None,
                 target_tokenizer=None,
                 download='infer',
                 random_split=True,
                 batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS,
                 ):
        super(OPUSDataModule, self).__init__()

        self.data_dir = data_dir
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_seq_len = max_seq_len
        self.download = download.lower()
        self.random_split = random_split
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

        if self.download == 'infer':
            if any(not os.path.exists(path) for path in [self.tmx_path]):
                self.download = 'yes'

        if self.download == 'yes':
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

    def train_test_val_split(self, data, train_percent, val_percent):

        # Calculate the sizes of each set
        num_samples = len(data)
        num_train = int(train_percent * num_samples)
        num_val = int(val_percent * num_samples)

        # Shuffle the indices to randomly select samples for each set
        indices = np.arange(num_samples)
        if self.random_split:
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

        train_data, test_data, val_data = self.train_test_val_split(data, 0.7, 0.1)

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

    from machine_translation.tokenizer import WordLevelTokenizer

    # Use txt for better interpretability
    source_tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer_source.txt')
    target_tokenizer_path = os.path.join(config.TOK_DIR, r'tokenizer_target.txt')

    # Check if a tokenizer backup exists
    if os.path.exists(source_tokenizer_path):
        print(f'Loading source tokenizer...')
        source_tokenizer = WordLevelTokenizer.load(source_tokenizer_path, driver='txt')
    else:
        source_tokenizer = None

    if os.path.exists(target_tokenizer_path):
        print(f'Loading target tokenizer...')
        target_tokenizer = WordLevelTokenizer.load(target_tokenizer_path, driver='txt')
    else:
        target_tokenizer = None

    datamodule = OPUSDataModule(
        config.DATA_DIR,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        download='infer',
        random_split=False
    )
    datamodule.prepare_data()  # Download the data
    datamodule.setup()  # Setup it

    # Check that the size of the tensors are right
    train_dataloader = datamodule.train_dataloader()

    for batch in train_dataloader:

        if source_tokenizer is not None and target_tokenizer is not None:
            encoder_input = batch['encoder_input']
            decoder_input = batch['decoder_input']
            encoder_mask = batch['encoder_mask']
            decoder_mask = batch['decoder_mask']
            label = batch['label']

            print(f'Encoder input (batch) size  : {encoder_input.shape}')
            print(f'Decoder input (batch) size  : {decoder_input.shape}')
            print(f'Encoder mask (batch) size   : {encoder_mask.shape}')
            print(f'Decoder mask (batch) size   : {decoder_mask.shape}')
            print(f'Label (batch) size          : {label.shape}')

        else:
            source_text = batch['source_text']
            target_text = batch['target_text']

            print(f'Source text: {source_text}')
            print(f'Target text: {target_text}')

        break
