import torch
from torch import nn
from model.basic.DeformConv2D import DeformableConv2d

class DeformEncoder(nn.Module):

    def __init__(self):
        super(DeformEncoder, self).__init__()

        self.cnn_b1 = nn.Sequential(
            DeformableConv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.cnn_b2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )

        self.cnn_b3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )

        self.cnn_b4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=2)
        )

    def forward(self, x):
        x = self.cnn_b1(x)
        x = self.cnn_b2(x)
        x = self.cnn_b3(x)
        x = self.cnn_b4(x)

        return x
