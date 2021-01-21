from .resnet import resnet20complex, resnet32complex, resnet44complex, resnet56complex, resnet110complex
from .alexnet import alexnetcomplex
from .vgg import vggcomplex
from .lenet import lenetcomplex
from .discriminator import Discriminator
from .complex_layers import RealToComplex, ComplexToReal

def load_complex_network(arch, num_classes, resnet_variant='alpha'):
    # Verbose
    if arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        print(f'Initializing {arch}-{resnet_variant} network...')
    elif arch in ['vgg', 'alexnet', 'lenet']:
        print(f'Initializing {arch} network...')
    else:
        raise NotImplementedError(f'{arch} is not an available network architecture.')

    # Initialize network
    if arch == 'resnet20':
        return resnet20complex(num_classes, resnet_variant)
    elif arch == 'resnet32':
        return resnet32complex(num_classes, resnet_variant)
    elif arch == 'resnet44':
        return resnet44complex(num_classes, resnet_variant)
    elif arch == 'resnet56':
        return resnet56complex(num_classes, resnet_variant)
    elif arch == 'resnet110':
        return resnet110complex(num_classes, resnet_variant)
    elif arch == 'vgg':
        return vggcomplex(num_classes)
    elif arch == 'alexnet':
        return alexnetcomplex(num_classes)
    elif arch == 'lenet':
        return lenetcomplex(num_classes)
