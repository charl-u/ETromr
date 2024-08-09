import torch
from torch import nn

class RNNDecoder(nn.Module):
    def __init__(self, rnn_hidden_units=512, rnn_hidden_layer=2, rnn_dropout=0.5, rnn_bidirectional=True):
        super(RNNDecoder, self).__init__()

        self.rnn_hidden_units = rnn_hidden_units
        self.rnn_hidden_layer = rnn_hidden_layer
        self.rnn_dropout = rnn_dropout
        self.rnn_bidirectional = rnn_bidirectional
        self.output_dim = 2 * rnn_hidden_units

        self.rnn = nn.LSTM(input_size=2048, hidden_size=self.rnn_hidden_units, num_layers=self.rnn_hidden_layer, dropout=self.rnn_dropout, bidirectional=self.rnn_bidirectional)
        

    def forward(self, x): 
        x, _ = self.rnn(x) # [width, batch, 1024]
        return x