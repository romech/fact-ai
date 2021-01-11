import torch
import torch.nn as nn
import numpy as np

def get_real_imag_parts(x):
    assert(x.size(1) == 2) # Complex tensor has real and imaginary parts in 2nd dim 
    return x[:,0], x[:,1]

def complex_norm(x):
    assert(x.size(1) == 2) # Complex tensor has real and imaginary parts in 2nd dim
    return torch.sqrt(x[:,0]**2 + x[:,1]**2)
    
class ActivationComplex(nn.Module):
    def __init__(self, c=1):
        super(ActivationComplex, self).__init__()
        assert(c>0)
        self.c = torch.Tensor([c])

    def forward(self, x):
        x_norm = complex_norm(x).unsqueeze(1)
        scale = x_norm/torch.maximum(x_norm, self.c)
        return x*scale

class MaxPool2dComplex(nn.Module):
    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        dilation=1,
        ceil_mode=False
    ):
        super(MaxPool2dComplex, self).__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True
        )

    def get_indice_elements(self, x, indices):
        ''' From: https://discuss.pytorch.org/t/pooling-using-idices-from-another-max-pooling/37209/4 '''
        x_flat = x.flatten(start_dim=2)
        output = x_flat.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        return output

    def forward(self, x):
        x_real, x_imag = get_real_imag_parts(x)
        x_norm = complex_norm(x) 

        # Max pool complex feature norms and get indices
        _, indices = self.pool(x_norm)  

        # Extract the matching real and imaginary components of the max pooling indices
        x_real = self.get_indice_elements(x_real, indices)
        x_imag = self.get_indice_elements(x_imag, indices)

        return torch.stack((x_real, x_imag), dim=1)

class DropoutComplex(nn.Module):
    def __init__(self, p):
        super(DropoutComplex, self).__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x_real, x_imag = get_real_imag_parts(x)

        # Apply dropout to real part
        x_real = self.dropout(x_real)

        # Drop the same indices in the imaginary part 
        # and scale the rest by 1/1-p 
        if self.train:
            mask = (x_real != 0).float()*(1/(1-self.p))
            x_imag *= mask

        return torch.stack((x_real, x_imag), dim=1)

class Conv2dComplex(nn.Module):
    ''' Section 3.2: https://arxiv.org/abs/1705.09792 '''
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        padding_mode='zeroes'
    ):
        super(Conv2dComplex, self).__init__()

        self.in_channels = in_channels

        self.conv_real = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding_mode=padding_mode,
            bias=False # Bias always false
        )
        self.conv_imag = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding_mode=padding_mode,
            bias=False # Bias always false
        )

    def forward(self, x):
        x_real, x_imag = get_real_imag_parts(x)
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return torch.stack((out_real, out_imag), dim=1)

class BatchNormComplex(nn.BatchNorm2d):
    ''' Adapted from: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L39 '''
    def __init__(
        self,
        num_features,
        momentum=0.1,
        track_running_stats=True
    ):
        super(BatchNormComplex, self).__init__(
            num_features=num_features,
            momentum=momentum,
            affine=False,
            track_running_stats=track_running_stats
        )

    def forward(self, x):
        # Setup exponential factor
        ema_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    ema_factor = 1.0/float(self.num_batches_tracked)

        # Calculate mean of complex norm
        x_norm = torch.pow(complex_norm(x), 2)
        if self.train:
            mean = x_norm.mean([0])
            with torch.no_grad():
                self.running_mean = ema_factor * mean + (1-ema_factor) * self.running_mean
        else:
            mean = self.running_mean

        # Normalize
        x /= torch.sqrt(mean[None, None, :, :, :])

        return x

class RealToComplex(nn.Module):
    def __init__(self, k):
        super(RealToComplex, self).__init__()
        self.k = k

    def forward(self, a, b):
        # Randomly choose theta
        theta = torch.tensor(np.random.choice(np.linspace(0, 2*np.pi)))

        # Convert to complex and rotate by theta
        real = a*torch.cos(theta) - b*torch.sin(theta)
        imag = b*torch.cos(theta) + a*torch.sin(theta)
        x = torch.stack((real, imag), dim=1)

        return x, theta

class Complex2Real(nn.Module):
    def __init__(self):
        super(Complex2Real, self).__init__()

    def forward(self, h, theta):
        # Apply opposite rotation to decode
        a, b = get_real_imag_parts(h)
        y = a*torch.cos(-theta) - b*torch.sin(-theta) # Only need real component
        
        return y

if __name__ == '__main__':  
    x = torch.rand((3,2,1,2,1))
    print(x)
    #print(x.size())

    #act = ActivationComplex()
    #y = act(x)

    #pool = MaxPool2dComplex(2, 2)
    #y = pool(x)

    #drop = DropoutComplex(0.5)
    #y = drop(x)

    norm = BatchNormComplex(1)
    y = norm(x)
    print(y)
    print(norm.running_mean)


    #print(y)
    #print(y.size())

    '''
    a = torch.ones(1,1,3,1)*3
    b = torch.ones(1,1,3,1)*5

    tocomplex = RealToComplex(k=10)
    toreal = Complex2Real()
    h, theta = tocomplex(a,b)
    y = toreal(h, theta)

    print(a)
    print(h)
    print(y)
    '''
