import torch
import torch.nn as nn
from torch.autograd import Variable


# input_dim, hidden_dim, layer_dim, output_dim
class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim = 1):
        super(LSTMModel, self).__init__()
        
        self.num_classes = output_dim
        self.num_layers = layer_dim
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size * self.num_layers, self.num_classes)
        print(self.fc)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size * self.num_layers)
        
        out = self.fc(h_out)
        
        return out
