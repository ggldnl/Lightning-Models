from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import nltk
import os


class WordLevelTokenizer:

    def __init__(self,
                 unk_token='[UNK]',
                 pad_token='[PAD]',
                 sos_token='[SOS]',
                 eos_token='[EOS]',
                 min_frequency=2
                 ):

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        self.unk_token_id = 0
        self.pad_token_id = 1
        self.sos_token_id = 2
        self.eos_token_id = 3

        self.special_tokens = [unk_token, pad_token, sos_token, eos_token]
        self.special_tokens_ids = [0, 1, 2, 3]

        self.min_frequency = min_frequency
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0

        nltk.download('punkt')

    @staticmethod
    def tokenize_sentence(sentence):
        tokens = [token.lower() for token in word_tokenize(sentence)]
        return tokens

    def train(self, corpus):

        # This is a simple tokenizer, no need to implement something fancy, we can use the split method
        # along with some basic cleaning procedure
        # all_tokens = [token.lower() for sentence in corpus for token in sentence.split()]
        all_tokens = [token.lower() for sentence in corpus for token in self.tokenize_sentence(sentence)]

        # Compute the frequency for each token
        token_frequency = {}
        for token in all_tokens:
            if token in token_frequency:
                token_frequency[token] += 1
            else:
                token_frequency[token] = 1

        # Remove tokens with low frequency
        unique_tokens = [token for token, freq in token_frequency.items() if freq >= self.min_frequency]

        # Populate word2index dictionary
        self.word2index = {word: idx + len(self.special_tokens) for idx, word in enumerate(sorted(unique_tokens))}

        # Add the special tokens
        for token, token_id in zip(self.special_tokens, self.special_tokens_ids):
            self.word2index[token] = token_id

        # Populate index2word dictionary
        self.index2word = {idx: word for word, idx in self.word2index.items()}

        self.vocab_size = len(self.word2index)

    def get_vocab_size(self):
        return self.vocab_size

    def token_to_id(self, token):
        return self.word2index[token]

    def id_to_token(self, token_id):
        return self.index2word[token_id]

    def sentence_to_tokens(self, sentence):
        sentence_tokens = [token.lower() for token in self.tokenize_sentence(sentence)]
        return [token if token in self.word2index else self.unk_token for token in sentence_tokens]

    def tokens_to_ids(self, tokens):
        return [self.word2index[token] for token in tokens]

    def encode(self, sentence):
        """
        Given a sentence, returns a list of token ids. The unknown token
        is used in place of unrecognized tokens.
        """

        # Convert the sentence into tokens and then into input ids
        sentence_tokens = self.sentence_to_tokens(sentence)
        sentence_ids = self.tokens_to_ids(sentence_tokens)

        return sentence_ids

    def decode(self, token_ids):
        """
        Given a list of token ids, returns a sentence.
        """
        return ' '.join([self.id_to_token(token_id) for token_id in token_ids])

    def to_pickle(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'wb') as file:
            pickle.dump({
                'vocab_size': self.vocab_size,
                'word2index': self.word2index,
                'index2word': self.index2word
            }, file)

    def from_pickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.vocab_size = data['vocab_size']
            self.word2index = data['word2index']
            self.index2word = data['index2word']

    def to_txt(self, path):

        # Get the directory part of the file path
        parent_folder = os.path.dirname(path)

        # Create the parent folders if they don't exist
        os.makedirs(parent_folder, exist_ok=True)

        with open(path, 'w') as file:
            file.write(f"vocab_size: {self.vocab_size}\n")
            for word, index in self.word2index.items():
                file.write(f"{word}\t{index}\n")

    def from_txt(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            vocab_size = int(lines[0].split(':')[1])
            self.vocab_size = vocab_size

            word2index = {}
            for line in lines[1:]:
                word, index = line.strip().split('\t')
                word2index[word] = int(index)
            self.word2index = word2index

            index2word = {index: word for word, index in word2index.items()}
            self.index2word = index2word

    @classmethod
    def load(cls, path, driver='pkl'):

        driver = driver.lower()

        if driver not in ['pkl', 'pickle', 'txt']:
            raise ValueError(f'Invalid driver: {driver}')

        tokenizer = WordLevelTokenizer()

        if driver == 'pkl' or driver == 'pickle':
            tokenizer.from_pickle(path)
        else:
            tokenizer.from_txt(path)

        return tokenizer


if __name__ == '__main__':

    # Configure logging level
    import logging
    logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)

    from models.transformer.machine_translation.data import OPUSDataModule
    import models.transformer.machine_translation.config as config

    def create_tokenizer(stage='source'):  # stage = ['source', 'target']

        # Build a datamodule WITHOUT tokenizers
        datamodule = OPUSDataModule(
            config.DATA_DIR,
            max_seq_len=config.MAX_SEQ_LEN,
            download=False,
            random_split=False
        )
        datamodule.prepare_data()  # Download the data
        datamodule.setup()  # Setup it

        # Take the corpus (we need an iterable)
        train_dataloader = datamodule.train_dataloader()
        corpus = []
        for batch in train_dataloader:
            # batch[f'{stage}_text'] is a list of strings
            corpus.extend(batch[f'{stage}_text'])

        # Use the corpus to train the tokenizer
        tokenizer = WordLevelTokenizer()
        tokenizer.train(corpus)

        return tokenizer

    # Create or get tokenizers
    source_tokenizer_path = r'tokenizers/tokenizer_source.txt'
    target_tokenizer_path = r'tokenizers/tokenizer_target.txt'

    # Check if a tokenizer backup exists
    if os.path.exists(source_tokenizer_path):
        print(f'Loading source tokenizer...')
        source_tokenizer = WordLevelTokenizer.load(source_tokenizer_path, driver='txt')
    # If not, create a monolingual dataset and train them
    else:
        print(f'Creating source tokenizer...')
        source_tokenizer = create_tokenizer('source')
        source_tokenizer.to_txt(source_tokenizer_path)

    if os.path.exists(target_tokenizer_path):
        print(f'Loading target tokenizer...')
        target_tokenizer = WordLevelTokenizer.load(target_tokenizer_path, driver='txt')
    else:
        print(f'Creating target tokenizer...')
        target_tokenizer = create_tokenizer('target')
        target_tokenizer.to_txt(target_tokenizer_path)
