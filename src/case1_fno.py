import torch.nn as nn
import torch
from neuralop.models import FNO1d
import torch.nn.functional as F

class MultistepPredictorFNO(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        modes,
        input_channel,
        fno_output_channel,
        output_dim,
        output_horizon,
    ):
        super().__init__()

        self.output_horizon = output_horizon

        self.fno = FNO1d(
            n_modes_height=modes,
            n_layers=num_layers,
            hidden_channels=hidden_size,
            in_channels=input_channel,
            out_channels=fno_output_channel,
        )

        self.head = nn.Sequential(
            nn.Linear(fno_output_channel, fno_output_channel),
            nn.GELU(),
            nn.Linear(fno_output_channel, output_dim),
        )

    def forward(self, x):
        """
        x: (batch, input_grid, input_channel)

        returns:
            (batch, output_horizon, output_dim)
        """
        # FNO1d expects (batch, channels, grid)
        y = self.fno(x.transpose(1, 2))   # (B, C, G_in)

        # Resize latent grid to desired multistep horizon
        if y.shape[-1] != self.output_horizon:
            y = F.interpolate(
                y,
                size=self.output_horizon,
                mode="linear",
                align_corners=False,
            )  # (B, C, G_out)

        y = y.transpose(1, 2)             # (B, G_out, C)

        out = self.head(y)                # (B, G_out, output_dim)
        return out