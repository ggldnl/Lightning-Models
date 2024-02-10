import pytorch_lightning as pl
import torch.nn as nn
import torch


class LayerNorm(pl.LightningModule):
    """
    We compute a mean and a variance for each item in the batch independently
    of the other items and then compute their new values by normalizing by
    their own mean and variance. We usually add additive and multiplicative
    constants that introduce some fluctuations in the data. The network
    will learn to tune these two parameters to introduce fluctuations
    when necessary.

    x_j_hat = (x_j - mu_j) * (sigma_j**2 + eps) ** (-1/2)

    eps is needed as if sigma happens to be zero or very close to it
    then x_j_hat will become very big and this is undesirable since
    we can represent numbers up to a certain precision, we need to
    avoid numbers too big or too small.
    """

    def __init__(self,
                 eps: float = 10**-6
                 ):

        super(LayerNorm, self).__init__()

        self.eps = eps

        # Define additive and multiplicative constants
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1))  # Additive

    def forward(self, x):

        # dim=-1: compute the dimension of the last dimension (layer)
        # keepdim=True: usually computing the mean removes the dimension
        # for which it is computed, but we want to keep it
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) * (std**2 + self.eps) ** (-1/2) + self.bias
