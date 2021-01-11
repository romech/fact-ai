import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

def get_real_imag_parts(x):
    '''
    Extracts the real and imaginary component tensors from a complex number tensor

    Input:
        x: Complex number tensor of size [b,2,c,h,w]
    Output:
        real component tensor of size [b,c,h,w]
        imaginary component tensor of size [b,c,h,w]
    '''
    assert(x.size(1) == 2) # Complex tensor has real and imaginary parts in 2nd dim 
    return x[:,0], x[:,1]

def complex_norm(x):
    '''
    Calculates the complex norm for each complex number in a tensor

    Input:
        x: Complex number tensor of size [b,2,c,h,w]
    Output:
        tensor of norm values of size [b,c,h,w]
    '''
    assert(x.size(1) == 2) # Complex tensor has real and imaginary parts in 2nd dim
    return torch.sqrt(x[:,0]**2 + x[:,1]**2)

class RealToComplex(nn.Module):
    '''
    Converts a real value tensor a into a complex value tensor x (Eq. 2). 
    Adds a fooling counterpart b and rotates the tensor by a random angle theta.
    Returns theta for later decoding.
    
    Shape:
        Input: 
            a: [b,c,h,w]
            b: [b,c,h,w]
        Output:
            x: [b,2,c,h,w]
            theta: [1]
    '''
    def __init__(self):
        super(RealToComplex, self).__init__()

    def forward(self, a, b):
        # Randomly choose theta
        theta = torch.FloatTensor(1).uniform_(0, 2*np.pi)

        # Convert to complex and rotate by theta
        real = a*torch.cos(theta) - b*torch.sin(theta)
        imag = b*torch.cos(theta) + a*torch.sin(theta)
        x = torch.stack((real, imag), dim=1)

        return x, theta

class Complex2Real(nn.Module):
    '''
    Decodes a complex value tensor h into a real value tensor y by rotating 
    by -theta (Eq. 3). 

    Shape:
        Input:
            h: [b,2,c,h,w]
            theta: [1]
        Output: [b,c,h,w] 
    '''
    def __init__(self):
        super(Complex2Real, self).__init__()

    def forward(self, h, theta):
        # Apply opposite rotation to decode
        a, b = get_real_imag_parts(h)
        y = a*torch.cos(-theta) - b*torch.sin(-theta) # Only need real component
        
        return y
  
class ActivationComplex(nn.Module):
    '''
    Complex activation function from Eq. 6.

    Args:
        c: Positive constant (>0) from Eq. 6. Default: 1
    Shape:
        Input: [b,2,c,h,w]
        Output: [b,2,c,h,w]
    '''
    def __init__(self, c=1):
        super(ActivationComplex, self).__init__()
        assert(c>0)
        self.c = torch.Tensor([c])

    def forward(self, x):
        x_norm = complex_norm(x).unsqueeze(1)
        scale = x_norm/torch.maximum(x_norm, self.c)
        return x*scale

class MaxPool2dComplex(nn.Module):
    ''' 
    Complex max pooling operation. Keeps the complex number feature with the maximum norm within 
    the window, keeping both the corresponding real and imaginary components.
    
    Args:
        kernel_size: size of the window
        stride: stride of the window. Default: kernel_size
        padding: amount of zero padding. Default: 0
        dilation: element-wise stride in the window. Default: 1
        ceil_mode: use ceil instead of floor to compute the output shape. Default: False
    Shape:
        Input: [b,2,c,h_in,w_in]
        Output: [b,2,c,h_out,w_out]
    '''
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
    '''
    Complex dropout operation. Randomly zero out both the real and imaginary 
    components of a complex number feature.

    Args:
        p: probability of an element being zeroed
    Shape:
        Input: [b,2,c,h,w]
        Output: [b,2,c,h,w] 
    '''
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
    ''' 
    Complex 2d convolution operation. Implementation the complex convolution from 
    https://arxiv.org/abs/1705.09792 (Section 3.2) and removes the bias term
    to preserve phase.

    Args:
        in_channels: number of channels in the input
        out_channels: number of channels produced in the output
        kernel_size: size of convolution window
        stride: stride of convolution. Default: 1
        padding: amount of zero padding. Default: 0
        padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        groups: number of blocked connections from input to output channels. Default: 1
    Shape:
        Input: [b,2,c,h_in,w_in]
        Output: [b,2,c,h_out,w_out]
    '''
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode='zeroes',
        groups=1,
    ):
        super(Conv2dComplex, self).__init__()

        self.in_channels = in_channels

        self.conv_real = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
            bias=False # Bias always false
        )
        self.conv_imag = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            groups=groups,
            bias=False # Bias always false
        )

    def forward(self, x):
        x_real, x_imag = get_real_imag_parts(x)
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return torch.stack((out_real, out_imag), dim=1)

class BatchNormComplex(nn.BatchNorm2d):
    ''' 
    Complex batch normalization from Eq. 7. Code adapted from 
    https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py#L39 
    
    Args:
        size: size of a single sample [c,h,w].
        momentum: exponential averaging momentum term for running mean.
                  Set to None for simple average. Default: 0.1
        track_running_stats: track the running mean for evaluation mode. Default: True
    Shape:
        Input: [b,2,c,h,w]
        Output: [b,2,c,h,w]
    '''
    def __init__(
        self,
        size,
        momentum=0.1,
        track_running_stats=True
    ):
        super(BatchNormComplex, self).__init__(
            num_features=size[0],
            momentum=momentum,
            affine=False,
            track_running_stats=track_running_stats
        )

        self.running_mean = torch.zeros(size)



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
                print(self.running_mean)
                print(mean)
                self.running_mean = ema_factor * mean + (1-ema_factor) * self.running_mean
        else:
            mean = self.running_mean

        # Normalize
        x /= torch.sqrt(mean[None, None, :, :, :])

        return x
        
        
if __name__ == '__main__':  
    # RealToComplex + Complex2Real
    a = torch.rand(10,5,3,7)
    b = torch.rand(10,5,3,7)
    tocomplex = RealToComplex()
    toreal = Complex2Real()
    h, theta = tocomplex(a,b)
    y = toreal(h, theta)
    assert(h.size()==(10,2,5,3,7))
    assert(torch.allclose(a,y))

    # Activation    
    x = torch.rand((3,2,3,4,4))
    act = ActivationComplex()
    y = act(x)
    assert(np.all(torch.ge(x,y).numpy()))

    # Max pool
    x = torch.rand((3,2,3,4,4))
    pool = MaxPool2dComplex(2, 2)
    y = pool(x)
    assert(y.size()==(3,2,3,2,2))

    # Dropout
    x = torch.rand((3,2,3,4,4))
    drop = DropoutComplex(0.5)
    y = drop(x)
    assert(y.size()==(3,2,3,4,4))
    assert(0 in y)
    
    x = torch.rand((3,2,3,4,4)) 
    drop.eval()
    y = drop(x)
    assert(0 not in y)

    # Normalization
    x = torch.rand((3,2,3,4,4)) 
    norm = BatchNormComplex((3,4,4))
    y = norm(x)
