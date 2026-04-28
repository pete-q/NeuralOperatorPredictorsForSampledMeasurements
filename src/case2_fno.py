import torch.nn as nn
import torch
from neuralop.models import FNO1d

class PredictorFNO(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        modes,
        input_channel,
        fno_output_channel,
        output_dim,
    ):
        super().__init__()

        self.fno = FNO1d(
            n_modes_height=modes,
            n_layers=num_layers,
            hidden_channels=hidden_size,
            in_channels=input_channel,
            out_channels=fno_output_channel,
        )

        # scalar attention score per horizon location
        self.attn = nn.Sequential(
            nn.Linear(fno_output_channel, fno_output_channel),
            nn.GELU(),
            nn.Linear(fno_output_channel, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(fno_output_channel, fno_output_channel),
            nn.GELU(),
            nn.Linear(fno_output_channel, output_dim),
        )

    def forward(self, x):
        """
        x: (batch, grid, input_channel)

        returns:
            (batch, output_dim)
        """
        # FNO1d expects (batch, channels, grid)
        y = self.fno(x.transpose(1, 2))      # (B, C, G)
        y = y.transpose(1, 2)                # (B, G, C)

        # attention weights over horizon
        scores = self.attn(y)                # (B, G, 1)
        weights = torch.softmax(scores, dim=1)

        # weighted sum over horizon
        pooled = (weights * y).sum(dim=1)    # (B, C)

        out = self.head(pooled)              # (B, output_dim)
        return out