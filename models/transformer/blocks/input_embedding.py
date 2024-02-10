import pytorch_lightning as pl
import torch.nn as nn
import math


class InputEmbeddings(pl.LightningModule):
    """
    Takes the input and converts it into an embedding.
    We start with the original sequence (list of tokens).
    We assign each token a numerical id that correspond
    to the position of the token in a vocabulary. Each
    numerical id corresponds to an embedding vector.
    """

    def __init__(self,
                 d_model: int,  # Embedding vector dimension (dimension of the embedding space)
                 vocab_size: int  # How many words are in the vocabulary
                 ):
        super(InputEmbeddings, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Given a number, it will provide the respective embedding vector.
        # This is a mapping between numbers and vectors, it yields always
        # the same result for a given number
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
