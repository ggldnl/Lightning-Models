import lightning as pl
import torch
import torch.nn as nn


class SelfAttention(pl.LightningModule):
    """
    Multi head self attention module

                          Multi-Head Attention
                                  ^
                                  |
                                  |
                            +----------+
                            |  Linear  |
                            +----------+
                                  ^
                                  |
                                  |
                            +----------+
                            |  Concat  |
                            +----------+
                                  ^
                                  |
                                  |
    +----------------------------------------------------------+
    |                                                          |+
    |               Scaled Dot-Product Attention               ||+
    |                                                          |||
    +----------------------------------------------------------+||
     +---^-----------------------^-----------------------^------+|
      +--|^----------------------|^----------------------|^------+
         ||^                     ||^                     ||^
         |||                     |||                     |||
         |||                     |||                     |||
    +----------+            +----------+            +----------+
    |  Linear  |+           |  Linear  |+           |  Linear  |+
    +----------+|+          +----------+|+          +----------+|+
     +----------+|           +----------+|           +----------+|
      +----------+            +----------+            +----------+
          ^                       ^                       ^
          |                       |                       |
          |                       |                       |
          V                       K                       Q

    """

    def __init__(self, embed_size: int, heads: int):
        """

        :param embed_size: Size of the embeddings
        :param heads: How many parts we split the embeddings into (multi-head attention).
        """

        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_size = embed_size // heads

        assert (self.head_size * heads == embed_size), f'Embedding size ({embed_size}) should be divisible by heads ({heads})'

        # Define the linear value, key and query layers

        # The query vector represents what we are looking for or focusing on in a sequence.
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)

        # The key vector provides context or background information about each element in the sequence.
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)

        # The value vector holds the actual information associated with each element in the sequence.
        self.values = nn.Linear(self.head_size, self.head_size, bias=False)

        self.fc_out = nn.Linear(self.heads * self.head_size, self.embed_size)  # heads * head_size == embed_size anyways

    def forward(self, values, keys, queries, mask):

        num_train_examples = queries.shape[0]

        # These values depend on where the self attention module is used (either encoder or decoder)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads pieces each of self.head_size size
        values = values.reshape(num_train_examples, value_len, self.heads, self.head_size)
        keys = keys.reshape(num_train_examples, key_len, self.heads, self.head_size)
        queries = queries.reshape(num_train_examples, query_len, self.heads, self.head_size)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # query is the target sequence, key is the source. The energy
        # will tell us, for each token in our target, how much we pay
        # attention to each token in the source.
        # queries shape: (num_train_examples, query_len, heads, heads_size) -> nqhd
        # keys shape: (num_train_examples, keys_len, heads, heads_size) -> nkhd
        # energy shape: (num_train_examples, heads,  query_len, keys_len) -> nhqk
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # If the element of the mask is 0, replace the corresponding energy value with -inf.
            # -inf values will be set to 0 by the softmax.
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Attention (Q, K, V) = softmax((Q * T^T) / sqrt(d_k)) * V
        # dim=3 -> Attention scores normalized to 1 across the source sequence.
        # This way, if a source token has 0.8 score, that means we are paying
        # 80% attention to that token.
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_size)
        # out shape: (N, query_len, heads, head_size)
        # key_len and value_len match
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # Concatenation, flatten last two dimensions
        out = out.reshape(num_train_examples, query_len, self.heads*self.head_size)

        # Send it through the last linear layer
        out = self.fc_out(out)

        return out
