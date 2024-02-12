import pytorch_lightning as pl
import torch.nn as nn
import math


class MultiHeadAttention(pl.LightningModule):
    #    Multi head self attention module
    #
    #                          Multi-Head Attention
    #                                  ^
    #                                  |
    #                                  |
    #                            +----------+
    #                            |  Linear  |
    #                            +----------+
    #                                  ^
    #                                  |
    #                                  |
    #                            +----------+
    #                            |  Concat  |
    #                            +----------+
    #                                  ^
    #                                  |
    #                                  |
    #    +----------------------------------------------------------+
    #    |                                                          |+
    #    |               Scaled Dot-Product Attention               ||+
    #    |                                                          |||
    #    +----------------------------------------------------------+||
    #     +---^-----------------------^-----------------------^------+|
    #      +--|^----------------------|^----------------------|^------+
    #         ||^                     ||^                     ||^
    #         |||                     |||                     |||
    #         |||                     |||                     |||
    #    +----------+            +----------+            +----------+
    #    |  Linear  |+           |  Linear  |+           |  Linear  |+
    #    +----------+|+          +----------+|+          +----------+|+
    #     +----------+|           +----------+|           +----------+|
    #      +----------+            +----------+            +----------+
    #          ^                       ^                       ^
    #          |                       |                       |
    #          V                       K                       Q
    #          |                       |                       |
    #          +-----------------------+-----------------------+
    #                                  |
    #                                  |
    #                                Input (in the encoder)

    def __init__(self,
                 d_model: int,
                 h: int,  # Number of heads
                 dropout: float
                 ):
        """
        We have our input sequence (seq_len, d_model). In the encoder, we split
        it into three matrices with size (seq_len, d_model) that are exactly the
        same as the input (things will be slightly different in the decoder), that
        are query, keys and values. We multiply these matrices by three weight
        matrices W_q, W_K, W_v with size (d_model, d_model). This results in a
        new matrix of size (seq_len, d_model). We then split this new matrix
        into h vectors (where h is the number of heads). We split these matrices
        along the embedding dimension, not along the sequence dimension. This means
        each head will have access to the full sequence but will have a different
        part of the embedding of each token. We apply the attention to these
        matrices that will return smaller matrices that we will concatenate back
        and multiply by another weight matrix to obtain the result.
        """

        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.dropout = nn.Dropout(dropout)

        self.d_k = d_model // h

        # self.w_q = nn.Linear(d_model, d_model, bias=False)
        # self.w_k = nn.Linear(d_model, d_model, bias=False)
        # self.w_v = nn.Linear(d_model, d_model, bias=False)
        # self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Just to visualize it
        self.attention_scores = None

    @staticmethod
    def attention(query, key, value, mask, dropout, spatial_encoding, edge_encoding, centrality):

        # Last dimension of the query, key and value
        d_k = query.shape[-1]

        # We are computing things one batch at a time but the formula
        # is defined on a single input, so we transpose the last two dimensions.
        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply the mask
        if mask is not None:
            # Replace all the values for which the statement "mask == 0" with -1e9
            attention_scores.masked_fill(mask == 0, -1e9)

        # output shape is (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Multiply the output of the softmax by the v matrix
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None, spatial_encoding, edge_encoding, centrality):
        """
        Mask is used when we want some tokens not to interact with some other tokens.

        The attention is defined as:

        Attention(Q, K, V) = softmax((QV^T) * (d_k) ** (-1/2)) * V

        The input to the softmax is the attention square matrix, that tells us the
        attention, for each token, to all the other tokens.

        If we want some tokens not to interact with some other tokens, we can apply
        a mask to the attention matrix before passing it through the softmax. The
        mask will replace the masked tokens with a very small value that will be
        transformed to 0 by the softmax.
        """

        query = self.w_q(q)  # shape (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)  # shape (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        values = self.w_v(v)  # shape (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        """
        Divide the query, key and values to smaller matrices and pass them to the heads.
        The view method returns a new tensor with the same data as the self tensor but
        of a different shape. We will split the last dimension (d_model) into h tensors
        with size d_k (split the embedding, not the sequence). We transpose since we
        want each head to watch a tensor of size (seq_len, d_k) which means each head
        will see the full sequence (seq_len) but only a small part of each embedding (d_k)
        """
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = values.view(values.shape[0], values.shape[1], self.h, self.d_k).transpose(1, 2)

        # attention_scores is the output of the softmax
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout, spatial_encoding, edge_encoding, centrality)

        # Revert the previous changes and concatenate the small matrices into a single matrix
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by w_o
        return self.w_o(x)
