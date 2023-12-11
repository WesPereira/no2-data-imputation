import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ConvLSTMModel(nn.Module):

    def __init__(
        self,
        convs,
        kernel_size,
        hidden_dim,
        layer_dim,
        output_dim = 1
    ):
        super(ConvLSTMModel, self).__init__()

        self.num_classes = output_dim
        self.num_layers = layer_dim
        self.hidden_size = hidden_dim

        conv_layers = []
        for i, (inp_sz, out_sz) in enumerate(list(zip(convs, convs[1:]))):
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

        self.lstm = nn.LSTM(input_size=convs[-1], hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size * self.num_layers, self.num_classes)


    def init_hidden(self, batch_size):
        h_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size))

        return h_0, c_0

    def forward(self, x):
        # Convolutional step
        out = torch.swapaxes(x, -1, -2)
        out = self.conv_layers(out)
        out = torch.swapaxes(out, -1, -2)

        # Init lstm hidden states
        h_0, c_0 = self.init_hidden(out.size(0))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(out, (h_0, c_0))

        # Fully-connected step
        h_out = h_out.view(-1, self.hidden_size * self.num_layers)
        out = F.tanh(h_out)
        out = self.fc(h_out)

        return out
