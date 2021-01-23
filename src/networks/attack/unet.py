import torch
import torch.nn as nn

class Block(nn.Module):
    ''' 
    Convolution block with 6 convolutions

    Args:
        in_channels: number of channels in input
        out_channels: number of channels in output
    Shape:
        Input: [b,in_channels,h,w]
        Output: [b,out_channels,h,w]
    '''
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    '''
    Maxpool + convolution block

    Args:
        in_channels: number of channels in input
        out_channels: number of channels in output
    Shape:
        Input: [b,in_channels,h,w]
        Output: [b,out_channels,h/2,w/2]
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    '''
    Bilinear upsample + skip connection concat + convolution block
    
    Args:
        in_channels: number of channels in stacked input (x + skip)
        out_channels: number of channels in output
    Shape:
        Input: 
            x: [b,in_channels/2,h,w]
            skip: [b,in_channels/2,2h,2w]
        Output: [b,out_channels,2h,2w]
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = Block(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNet(nn.Module):
    '''
    UNet for reconstructing the input image from intermediate 
    features (inversion attack 2). Takes input of any height and
    width and 

    Args:
        in_channels: number of channels in input
        out_size: target output size (h,w)
    Shape:
        Input: [b,in_channels,h,w]
        Output: [b,3,size[0],size[1]]
    '''
    def __init__(self, in_channels, out_size):
        super(UNet, self).__init__()

        self.upsample = nn.Upsample(size=out_size)

        # Encoder
        self.encoder1 = Block(in_channels, 64)
        self.encoder2 = DownBlock(64, 128)
        self.encoder3 = DownBlock(128, 256)
        self.encoder4 = DownBlock(256, 512)
        self.encoder5 = DownBlock(512, 512)

        # Decoder
        self.decoder1 = UpBlock(1024, 256)
        self.decoder2 = UpBlock(512, 128)
        self.decoder3 = UpBlock(256, 64)
        self.decoder4 = UpBlock(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        # Upsample input to match target output size
        x = self.upsample(x)

        # Encoder
        e1 = self.encoder1(x) 
        e2 = self.encoder2(e1) 
        e3 = self.encoder3(e2) 
        e4 = self.encoder4(e3) 
        e5 = self.encoder5(e4)        
        
        # Decoder
        out = self.decoder1(e5, e4)
        out = self.decoder2(out, e3)
        out = self.decoder3(out, e2)
        out = self.decoder4(out, e1)
        out = self.out_conv(out)
        return out


if __name__ == '__main__':
    import numpy as np
    size = tuple(np.random.randint(low=1, high=32, size=4))
    x = torch.randn(size)
    net = UNet(size[1], (32,32))
    out = net(x)
    assert(out.shape == (size[0],3,32,32))
