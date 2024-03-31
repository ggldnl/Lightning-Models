import torch.nn as nn
import torch


class ProjectionLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 vocab_size: int
                 ) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # applying the log Softmax function to the output
        return torch.log_softmax(self.proj(x), dim=-1)
