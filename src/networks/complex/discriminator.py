import torch
import torch.nn as nn

class Discriminator(nn.Module):
    '''
    Adversarial discriminator network.

    Args:
        size: Tuple of input shape (c,h,w)
    Shape:
        Input: [b,c,h,w]
        Output: [b,1]
    '''
    def __init__(self, size):
        super(Discriminator, self).__init__()
        in_channels = size[0]

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2*size[0]*size[1]//2*size[2]//2, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == '__main__':
    x = torch.randn((4,4,32,32))
    d = Discriminator((4,32,32))
    y = d(x)
    assert(y.size(0) == 4 and y.size(1) == 1)

