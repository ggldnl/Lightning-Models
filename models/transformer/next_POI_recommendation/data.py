from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import config
import torch
import utils
import os


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0


class FoursquareDataset(Dataset):

    def __init__(self, data, max_seq_len, source_tokenizer=None, target_tokenizer=None):
        """
        The transformer is a sequence-to-sequence model used for translation
        from a language to another. The two languages might use a different
        set of tokens. Input to this Dataset are the POI sequences
        """

        super(FoursquareDataset, self).__init__()

        self.data = data
        self.max_seq_len = max_seq_len
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        # Take the special tokens from the tokenizers (just to simplify the code)
        if source_tokenizer is not None:
            self.source_pad_token = self.source_tokenizer.pad_token_id
            self.source_sos_token = self.source_tokenizer.sos_token_id
            self.source_eos_token = self.source_tokenizer.eos_token_id
            self.source_msk_token = self.source_tokenizer.msk_token_id

        if target_tokenizer is not None:
            self.target_pad_token = self.target_tokenizer.pad_token_id
            self.target_sos_token = self.target_tokenizer.sos_token_id
            self.target_eos_token = self.target_tokenizer.eos_token_id
            self.target_msk_token = self.target_tokenizer.msk_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        source_sequence = self.data[item]
        target_sequence = [token for token in source_sequence]

        if self.source_tokenizer is None or self.target_tokenizer is None:
            return {
                "source_sequence": source_sequence,
                "target_sequence": target_sequence
            }

        source_token_ids = self.source_tokenizer.encode(source_sequence)
        target_token_ids = self.target_tokenizer.encode(target_sequence)

        enc_padding_tokens = self.max_seq_len - len(source_token_ids) - 2
        dec_padding_tokens = self.max_seq_len - len(target_token_ids) - 1

        assert enc_padding_tokens > 0, (f"Encoder input sequence is too long [{enc_padding_tokens}]. Try increasing "
                                        f"the maximum sequence length. Sequence: {source_sequence}.")

        assert dec_padding_tokens > 0, (f"Decoder input sequence is too long [{dec_padding_tokens}]. Try increasing "
                                        f"the maximum sequence length. Sequence: {target_sequence}.")

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
            "source_sequence": source_sequence,
            "target_sequence": target_sequence
        }


class FoursquareDataModule(LightningDataModule):
    """
    Custom PyTorch Lightning DataModule class. The datamodule will
    download the content at the url only if the required file does
    not exist. This datamodule implements the logic to handle the
    Foursquare dataset.

    This dataset contains check-ins in NYC and Tokyo collected
    for about 10 month (from 12 April 2012 to 16 February 2013).

    More information here:
    https://sites.google.com/site/yangdingqi/home/foursquare-dataset#h.p_ID_46
    """

    def __init__(self,
                 data_dir,
                 max_seq_len,
                 min_seq_len,
                 use='both',  # This string can either be 'nyc', 'tky', 'both'
                 source_tokenizer=None,
                 target_tokenizer=None,
                 download=False,
                 batch_size=config.BATCH_SIZE,
                 num_workers=config.NUM_WORKERS
                 ):
        super(FoursquareDataModule, self).__init__()

        if use not in ['both', 'nyc', 'tky']:
            raise ValueError(f'Value for the \'use\' parameter can only be \'nyc\', \'tky\' or \'both\' to use' +
                             'data from New York, Tokyo or both of them.')

        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.use = use.lower()
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.resource_url = r'http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip'
        self.nyc_path = os.path.join(self.data_dir, 'dataset_tsmc2014/dataset_TSMC2014_NYC.txt')
        self.tky_path = os.path.join(self.data_dir, 'dataset_tsmc2014/dataset_TSMC2014_TKY.txt')
        self.schema = [
            'User ID',
            'Venue ID',
            'Venue category ID',
            'Venue category name',
            'Latitude',
            'Longitude',
            'Timezone',
            'UTC time'
        ]

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def prepare_data(self):

        if self.download:

            zip_path = os.path.join(self.data_dir, 'data.zip')
            utils.download_resource(self.resource_url, zip_path)
            utils.extract_zip(zip_path, self.data_dir)

    @staticmethod
    def train_text_val_split(data, train_percent, val_percent):

        # Compute the sizes of each set
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

        # Create lists for each set using the selected indices
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        return train_data, test_data, val_data

    @staticmethod
    def filter_data(
            data_source,
            min_user_count_per_POI=10,
            min_POI_count_per_user=20,
            max_POI_count_per_user=50,
    ):

        # Filter POIs with unique visitors count less than min_user_count_per_POI
        venue_user_counts = data_source.groupby('Venue ID')['User ID'].nunique().reset_index(name='user_count')

        # Filter Venue IDs with user counts less than min_user_count_per_POI
        selected_venues = venue_user_counts[venue_user_counts['user_count'] >= min_user_count_per_POI]

        # Extract relevant rows from the original NYC dataset for the selected venues
        filtered_data = data_source[data_source['Venue ID'].isin(selected_venues['Venue ID'])]
        # Group by User ID and count unique Venue IDs
        user_visit_counts = filtered_data.groupby('User ID')['Venue ID'].nunique().reset_index(name='visit_count')

        # Filter users with visit counts between min_POI_count_per_user and max_POI_count_per_user
        selected_users = user_visit_counts[(user_visit_counts['visit_count'] >= min_POI_count_per_user) &
                                           (user_visit_counts['visit_count'] <= max_POI_count_per_user)]

        # Extract relevant rows from the original dataset for the selected users
        filtered_data = filtered_data[filtered_data['User ID'].isin(selected_users['User ID'])]

        # Sort it
        filtered_data['UTC time'] = pd.to_datetime(filtered_data['UTC time'], format='%a %b %d %H:%M:%S +0000 %Y')

        return filtered_data.sort_values(by=['User ID', 'UTC time'], ascending=[True, True])

    @staticmethod
    def create_user_POI_sequence_dict(data):

        # Initialize dictionaries to store sequences
        Lu_dict = {}

        # Iterate through the sorted DataFrame to create sequences
        for user_id, group in data.groupby('User ID'):
            Lu_sequence = [(venue_id, str(timestamp)) for venue_id, timestamp in
                           zip(group['Venue ID'], group['UTC time'])]

            Lu_dict[user_id] = Lu_sequence

        return Lu_dict

    @staticmethod
    def create_POI_sequence_list(data_dict):
        return [seq for user, seq in data_dict.items()]

    def split_sequences_exceeding_len(self, data):
        result_list = []
        for sublist in data:
            if len(sublist) > self.max_seq_len:

                # Split the sublist into sublists with a maximum length
                splitted_sublist = [sublist[i:i + self.max_seq_len] for i in range(0, len(sublist), self.max_seq_len)]

                # Add all the resulting sublists except the last one, for which we need to check the length
                result_list.extend(splitted_sublist[:-1])

                if len(splitted_sublist[-1]) > self.min_seq_len:
                    result_list.append(splitted_sublist[-1])
            else:

                # If the sublist is within the maximum length, add it as is
                result_list.append(sublist)

        return result_list

    def setup(self, stage=None):

        # raw_data contains the entries in the input files:
        # 'User ID',
        # 'Venue ID',
        # 'Venue category ID',
        # 'Venue category name',
        # 'Latitude',
        # 'Longitude',
        # 'Timezone',
        # 'UTC time'
        # data will instead contain the POI sequences for each user

        if self.use == 'nky':
            raw_data = utils.read_tsv(self.nyc_path, skip_header=False, encoding='latin-1')
        elif self.use == 'tky':
            raw_data = utils.read_tsv(self.tky_path, skip_header=False, encoding='latin-1')
        else:  # self.use == 'both'
            nyc_data = utils.read_tsv(self.nyc_path, skip_header=False, encoding='latin-1')
            tky_data = utils.read_tsv(self.tky_path, skip_header=False, encoding='latin-1')
            raw_data = nyc_data + tky_data

        # Create a dataframe (this will simplify things later)
        df = pd.DataFrame(raw_data, columns=self.schema)

        # Filter the data
        filtered_df = self.filter_data(df)

        # Build the sequences
        data_dict = self.create_user_POI_sequence_dict(filtered_df)

        # Discard the user and create a list of POI sequences
        data = self.create_POI_sequence_list(data_dict)

        # Split sequences that exceed max length
        data = self.split_sequences_exceeding_len(data)

        train_data, test_data, val_data = self.train_text_val_split(data, 0.7, 0.1)

        self.train_dataset = FoursquareDataset(
            train_data,
            self.max_seq_len,
            self.source_tokenizer,
            self.target_tokenizer
        )

        self.test_dataset = FoursquareDataset(
            test_data,
            self.max_seq_len,
            self.source_tokenizer,
            self.target_tokenizer
        )

        self.val_dataset = FoursquareDataset(
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
            collate_fn=lambda x: x
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda x: x
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda x: x
        )


if __name__ == '__main__':

    datamodule = FoursquareDataModule(
        config.DATA_DIR,
        config.MAX_SEQ_LEN,
        config.MIN_SEQ_LEN,
        use='both',
        download=False
    )

    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        elem = batch[0]
        source_text = elem['source_sequence']
        target_text = elem['target_sequence']
        print(f'Batch size: {len(source_text)}')
        print(f'Source:\n{source_text}')
        print(f'Target:\n{target_text}')
        break
