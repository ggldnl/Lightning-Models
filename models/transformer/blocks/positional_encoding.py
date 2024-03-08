import pytorch_lightning as pl
import torch.nn as nn
import torch
import math


class PositionalEncoding(pl.LightningModule):
    """
    Each sequence is mapped to a list of vectors by the embedding layer.
    We need to convey to the model information on the position of each
    token inside the sequence. To do so, to each token we sum another
    embedding vector of the same size that encodes the position of the
    token in the sequence. The sequences have a fixed maximum length,
    so we can compute the position embedding vectors once and reuse
    them for every sequence during training and inference.
    """

    def __init__(self,
                 d_model: int,  # Size of the embedding space = size of the position embedding vectors
                 seq_len: int,  # Maximum length of the sequence
                 dropout: float
                 ):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        """
        On the paper, the formula used to compute the position embedding is the following:
        
        PE(pos, 2i) = sin(pos / (10000 ^ (2i / d_model)))
        PE(pos, 2i+1) = cos(pos / (10000 ^ (2i / d_model)))
        
        We implement the same formula in log space to increase numerical stability.

        Numerical stability refers to how a malformed input affects the execution 
        of an algorithm and to the goodness of an implementation of a function. 
        We can't implement a function perfectly in an algorithm due to various 
        approximation errors introduced along the way (how we store numbers and 
        so on). Our implementation of a function will take the same input value 
        and will map it to some other value with respect to the expected value 
        (f_impl(x) != f(x)). If the function we are implementing is very sensitive 
        to small changes (it's conditioning is bad -> ill-conditioned) then a small 
        change in the input may result in a large change in the output.
        """

        # We need a vector of seq_len containing embeddings of size d_model
        position_embedding = torch.zeros(seq_len, d_model)

        # Create a vector of size seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # Apply sin to even positions
        position_embedding[:, 0::2] = torch.sin(position * denom)

        # Apply cos to odd positions
        position_embedding[:, 1::2] = torch.cos(position * denom)

        # We will have a batch of sequences so we need to add a new dimension
        position_embedding = position_embedding.unsqueeze(0)  # Shape (1, seq_len, d_model)

        """
        We register this tensor in the buffer of the model. This is done when we have
        a tensor that we want to store in the model not as learned parameter but as 
        constant 
        """
        self.register_buffer('pe', position_embedding)

    def forward(self, x):
        if len(x.shape) > 2:  # We are training, we have a batch dimension
            # We take the dimension 1 since we need to sum the positional encoding to the token dimension
            x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        else:  # We are making inference, we don't have a batch dimension
            x = x + (self.pe[:, :]).requires_grad_(False)
        return self.dropout(x)
