import torch
import torch.nn as nn


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.Linear = nn.ModuleList()
        for i in range(self.channels):
            self.Linear.append(nn.Linear(self.seq_len, self.pred_len))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :]
        x = x - seq_last
        output = torch.zeros(
            [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
        ).type_as(x)
        for i in range(self.channels):
            output[:, :, i] = self.Linear[i](x[:, :, i])
        x = output
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
