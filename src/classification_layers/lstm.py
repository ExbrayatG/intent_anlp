import torch
import torch.nn as nn


class LSTM(nn.module):
    def __init__(self, D_in, D_out, H) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=D_in, hidden_size=H, num_layers=1, batch_first=True
        )
