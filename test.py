from src.networks.complex.resnet import resnet20complex
from src.networks.complex.complex_layers import RealToComplex, ComplexToReal

e, p, d = resnet20complex(20)
r2c = RealToComplex()
c2r = ComplexToReal()

import torch
import numpy as np
x1 = torch.randn((4,3,2,1))
indices = np.random.permutation(x1.size(0))
a = e(x1)
print(a.size())
x, theta = r2c(a,a[indices])
h = p(x)
y = c2r(h, theta)
out = d(y)
print(out.size())
print(a)
print(out)
