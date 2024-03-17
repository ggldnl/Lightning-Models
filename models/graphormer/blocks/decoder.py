import pytorch_lightning as pl

from models.transformer_legacy.blocks.layer_norm import LayerNorm


class Decoder(pl.LightningModule):

    def __init__(self, layers):
        """
        The decoder is made up of many decoder blocks, we can have up to N of them
        """

        super(Decoder, self).__init__()

        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, source_mask, target_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        return self.norm(x)
