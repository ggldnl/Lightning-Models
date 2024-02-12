import pytorch_lightning as pl

from models.transformer.blocks.layer_norm import LayerNorm


class Encoder(pl.LightningModule):
    #                                     |
    #                 +--------------------------------------+
    #                 |                   |                  |
    #                 |                  ...                 |
    #                 |                                      |
    #            Nx   |                Encoder               |
    #                 |                 Block                |
    #                 |                                      |
    #                 |                  ...                 |
    #                 |                   |                  |
    #                 +--------------------------------------+
    #                                     |

    def __init__(self, layers):
        """
        The encoder is made up of many encoder blocks, we can have up to N of them
        """

        super(Encoder, self).__init__()

        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
