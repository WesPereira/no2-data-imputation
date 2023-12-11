from typing import List

import torch
import torch.nn as nn


class Conv1DModel(nn.Module):
    def __init__(
        self,
        convs: List[int],
        linears: List[int],
        kernel_size: int
    ):
        super(Conv1DModel, self).__init__()

        conv_layers = []
        for inp_sz, out_sz in list(zip(convs, convs[1:])):
            conv_layers.append(
                nn.Conv1d(inp_sz, out_sz, kernel_size)
            )
            conv_layers.append(
                nn.BatchNorm1d(out_sz)
            )
            conv_layers.append(
                nn.ReLU()
            )
            conv_layers.append(
                nn.MaxPool1d(2)
            )
        self.conv_layers = nn.Sequential(*conv_layers)

        self.fc1 = nn.Linear(linears[0], linears[1])
        self.fc2 = nn.Linear(linears[1], linears[2])

    def forward(self, x):
        out = torch.swapaxes(x, -1, -2)
        out = self.conv_layers(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
