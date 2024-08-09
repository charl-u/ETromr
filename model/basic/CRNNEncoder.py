from torch import nn
from model.basic.CNNEncoder import CNNEncoder
from model.basic.RNNDecoder import RNNDecoder

class CRNNEncoder(nn.Module):
    def __init__(self,
                    cnn: nn.Module,
                    rnn: nn.Module 
                ):
        super(CRNNEncoder, self).__init__()
        self.cnn = CNNEncoder()
        self.rnn = RNNDecoder()

    def forward(self, x):
        x = self.cnn(x)
        x = self.rnn(x)
        return x