import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=32, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)  # [T,H]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)  # [T,H] => [1,T,H]
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)  # [T,H] => [1,T,H] => [T,1,H]
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, : x.size(1), :]  # x[B,T,H] + pe[1,0:T,H]
        else:
            x = x + self.pe[: x.size(0), :, :]  # x[T,B,H] + pe[0:T,1,H]
        return self.dropout(x)