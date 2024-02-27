import pickle
import os


class POISequenceTokenizer:

    def __init__(self,
                 unk_token='[UNK]',
                 pad_token='[PAD]',
                 sos_token='[SOS]',
                 eos_token='[EOS]',
                 msk_token='[MSK]',
                 min_frequency=2
                 ):

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.msk_token = msk_token

        self.unk_token_id = 0
        self.pad_token_id = 1
        self.sos_token_id = 2
        self.eos_token_id = 3
        self.msk_token_id = 4

        self.special_tokens = [unk_token, pad_token, sos_token, eos_token, msk_token]
        self.special_tokens_ids = [i for i in range(len(self.special_tokens))]

        self.min_frequency = min_frequency
        self.word2index = {}
        self.index2word = {}
        self.vocab_size = 0

    def train(self, sequences):

        all_tokens = [poi for sequence in sequences for poi in sequence]

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
        self.word2index = {word: idx + len(self.special_tokens) + 1 for idx, word in enumerate(unique_tokens)}

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

    def sequence_to_tokens(self, sentence):
        return [token if token in self.word2index else self.unk_token for token in sentence]

    def tokens_to_ids(self, tokens):
        return [self.word2index[token] for token in tokens]

    def encode(self, sequence):

        # Convert the sequence into tokens and then into input ids
        sequence_tokens = self.sequence_to_tokens(sequence)
        sequence_ids = self.tokens_to_ids(sequence_tokens)

        return sequence_ids

    def save(self, path):

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

    def restore(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.vocab_size = data['vocab_size']
            self.word2index = data['word2index']
            self.index2word = data['index2word']

    @classmethod
    def load(cls, path):
        tokenizer = WordLevelTokenizer()
        tokenizer.restore(path)
        return tokenizer
