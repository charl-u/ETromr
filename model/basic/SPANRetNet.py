import sys
sys.path.append('/home/data/lyp_data/my_omr_projects/MyOMR')

from model.basic.SpanEncoder import ConvBlock, DSCBlock
from model.basic.RetNet import MyRetNetDecoder
from torch import nn
import torch
from einops import einsum, rearrange, repeat

class RetentionSpanEncoder(nn.Module):

    def __init__(self, in_channels, use_pe=False, dropout=0.4):
        super(RetentionSpanEncoder, self).__init__()

        self.use_pe = use_pe
        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=32, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=256, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=256, out_c=256, stride=(2,2), dropout=dropout)
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=256, out_c=256, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=256, out_c=256, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=256, out_c=256, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=256, out_c=256, stride=(1,1), dropout = dropout)
        ])

        self.pos_cenc = PositionalEncoding2D(256, 100, 100)

    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt
        
        if self.use_pe:
            x = self.pos_cenc(x)

        # x.permute(0, 1, 3, 2)
        # x = rearrange(x, 'b c h w -> (h w) b c')
        x = rearrange(x, 'b c h w -> b w (c h)')
        return x
    
class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max), device='cuda', requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class Model(nn.Module):
    def __init__(self, 
                in_channels=1, 
                encoder_dropout=0.4,
                 num_classes=88, 
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 decoder_dropout=0.1,
                 activation='swish',
                 dim_feedforward=2048,
                 norm_first=True,
                 layer_norm_eps=1e-5):
        super(Model, self).__init__()
        self.encoder = RetentionSpanEncoder(in_channels=in_channels, dropout=encoder_dropout)
        self.decoder = MyRetNetDecoder(
            input_dim=512,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=decoder_dropout,
            activation=activation,
            dim_feedforward=dim_feedforward,
            norm_first=norm_first,
            layer_norm_eps=layer_norm_eps
        )

        self.out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.out(x)
        return x
    
    @torch.no_grad()
    def generate():
        pass

if __name__ == '__main__':
    # model = Model(
    #     in_channels=1,
    #     encoder_dropout=0.4,
    #     num_classes=88,
    #     d_model=512,
    #     nhead=8,
    #     num_layers=6,
    #     decoder_dropout=0.1,
    #     activation='swish',
    #     dim_feedforward=2048,
    #     norm_first=True,
    #     layer_norm_eps=1e-5
    # )

    # x = torch.randn(1, 1, 128, 1024)
    # y = model(x)
    # print(y.shape)

    model = PositionalEncoding2D(dim=2, h_max=5, w_max=5)
    x = torch.ones((1, 2, 3, 3))
    y = model(x)
    print(y.shape)