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
    assert(x.size(1) == 2) # Complex tensor has real and imaginary components in 2nd dim 
    return x[:,0], x[:,1]

def complex_norm(x):
    '''
    Calculates the complex norm for each complex element in a tensor

    Input:
        x: Complex number tensor of size [b,2,c,h,w]
    Output:
        tensor of norm values of size [b,c,h,w]
    '''
    assert(x.size(1) == 2) # Complex tensor has real and imaginary components in 2nd dim
    x_real, x_imag = get_real_imag_parts(x)
    x_real = x_real.clone()
    x_imag = x_imag.clone()
    return torch.sqrt(torch.pow(x_real, 2) + torch.pow(x_imag, 2) + 1e-5)

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
        theta = a.new(1).uniform_(0, 2*np.pi)

        # Convert to complex and rotate by theta
        real = a*torch.cos(theta) - b*torch.sin(theta)
        imag = b*torch.cos(theta) + a*torch.sin(theta)
        x = torch.stack((real, imag), dim=1)

        return x, theta

class ComplexToReal(nn.Module):
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
        super(ComplexToReal, self).__init__()

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
        c = self.c.to(x.device)
        scale = x_norm/torch.maximum(x_norm, c)
        return x*scale

def activation_complex(x, c):
    '''
    Complex activation function from Eq. 6. This is a functional api to 
    use in networks that don't have a static c value (AlexNet, LeNet, etc.).

    Input:
        x: Complex number tensor of size [b,2,c,h,w]
        c: Positive constant (>0) from Eq. 6.
    Output:
        output tensor of size [b,2,c,h,w]
    '''
    assert(c>0)
    x_norm = complex_norm(x).unsqueeze(1)
    c = torch.Tensor([c]).to(x.device)
    scale = x_norm/torch.maximum(x_norm, c)
    return x*scale

def activation_complex_dynamic(x):
    '''
    Complex activation function from Eq. 6. This is a functional api to
    use in networks that don't have a static c value (AlexNet, LeNet, etc.).

    Input:
        x: Complex number tensor of size [b,2,c,h,w] or [b,2,f]
    Output:
        output tensor of size [b,2,c,h,w] or [b,2,f]
    '''
    x_norm = complex_norm(x)
    if x.dim() == 5:
        # for [b,2,c,h,w] inputs
        scale = x_norm.unsqueeze(1)/torch.maximum(x_norm.unsqueeze(1),
                                                  x_norm.mean((2, 3))[:, :, None, None].unsqueeze(1))
    else:
        # for [b,2,f] inputs
        scale = x_norm.unsqueeze(1)/torch.maximum(x_norm.unsqueeze(1),
                                                  x_norm.mean(1)[:, None, None])

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
        if self.training:
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
        padding_mode='zeros',
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

class LinearComplex(nn.Module):
    '''
    Complex linear layer. The bias term is removed in order to leave the phase invariant.

    Args:
        in_features: number of features of the input
        out_features: number of channels of the produced output
    Shape:
        Input: [b,2,in_features]
        Output: [b,2,out_features]
    '''
    def __init__(self, in_features, out_features):
        super(LinearComplex, self).__init__()

        self.linear_real = nn.Linear(in_features, out_features, bias=False)
        self.linear_imag = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x_real, x_imag = get_real_imag_parts(x)
        out_real = self.linear_real(x_real) - self.linear_imag(x_imag)
        out_imag = self.linear_real(x_imag) + self.linear_imag(x_real)
        return torch.stack((out_real, out_imag), dim=1)

class BatchNormComplex(nn.Module):
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
        momentum=0.1,
        track_running_stats=True
    ):
        super(BatchNormComplex, self).__init__()

        self.track_running_stats = track_running_stats
        self.num_batches_tracked = 0
        self.momentum = momentum
        self.running_mean = 0

    def forward(self, x):
        if self.track_running_stats == False:
            # Calculate mean of complex norm
            x_norm = torch.pow(complex_norm(x), 2)
            mean = x_norm.mean([0])
        else:
            # Setup exponential factor
            ema_factor = 0.0
            if self.training and self.track_running_stats:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                    if self.momentum is None:
                        ema_factor = 1.0/float(self.num_batches_tracked) # Cumulative moving average
                    else:
                        ema_factor = self.momentum

            # Calculate mean of complex norm
            if self.training :
                x_norm = torch.pow(complex_norm(x), 2)
                mean = x_norm.mean([0])
                with torch.no_grad():
                    self.running_mean = ema_factor * mean + (1-ema_factor) * self.running_mean
            else:
                if type(self.running_mean) == int:
                    mean = x.new(x.size(2), x.size(3), x.size(4))*0
                else:
                    mean = self.running_mean

        # Normalize
        x /= torch.sqrt(mean[None, None, :, :, :] + 1e-5)
        return x
        
        
if __name__ == '__main__':  
    # RealToComplex + Complex2Real
    for i in range(100):
        a = torch.rand(10,5,3,7)
        b = torch.rand(10,5,3,7)
        tocomplex = RealToComplex()
        toreal = ComplexToReal()
        h, theta = tocomplex(a,b)
        y = toreal(h, theta)
        assert(h.size()==(10,2,5,3,7))
        assert(torch.allclose(a,y, atol=1e-07))

    # Activation    
    x = torch.rand((1,2,1,4,4)).cuda()
    act = ActivationComplex()
    y = act(x)
    assert(np.all(torch.ge(x,y).numpy()))
    y2 = activation_complex(x, c=1)
    assert(np.all(torch.eq(y,y2).numpy()))

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
    
    x = torch.rand((1,2,1,4,4)) 
    x_clone = x.clone()
    drop.eval()
    y = drop(x)
    assert(0 not in y)
    assert(np.all(torch.eq(x_clone,y).numpy()))

    # Normalization
    norm = BatchNormComplex()
    for i in range(100):
        x = torch.rand((3,2,3,4,4)) 
        mean_before = norm.running_mean
        y = norm(x)
        mean_after = norm.running_mean
        if i>0:
            assert(not np.any(torch.eq(mean_before, mean_after).numpy())) # Running mean is being updated

    norm.eval()
    x = torch.rand((3,2,3,4,4)) 
    mean_before = norm.running_mean
    y = norm(x)
    mean_after = norm.running_mean
    assert(np.all(torch.eq(mean_before, mean_after).numpy())) # Running mean is not updated during eval

    # Conv
    x = torch.rand((3,2,3,4,4)) 
    conv = Conv2dComplex(3, 5, 2, stride=2)
    y = conv(x)
    assert(y.size() == (3,2,5,2,2))