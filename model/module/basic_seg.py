from torch import nn

__all__ = ['_FCNHead']


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__(**kwargs)
        self.block = list()
        inter_channels = in_channels // 4
        self.block.append(nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False))
        self.block.append(norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.block.append(nn.ReLU(inplace=True))
        self.block.append(nn.Dropout(0.1))
        self.block.append(nn.Conv2d(inter_channels, channels, kernel_size=1))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)
