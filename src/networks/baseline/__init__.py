from .alexnet import alexnet
from .lenet import lenet
from.vgg import vgg
from .resnet import resnet20, resnet32, resnet44, resnet56, resnet110

def load_baseline_network(arch, num_classes, resnet_variant='alpha', additional_layers=False):
    # Verbose
    if arch in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        print(f'Initializing {arch}-{resnet_variant} network...')
        if additional_layers: print('\tUsing additional layers')
    elif arch in ['vgg', 'alexnet', 'lenet']:
        print(f'Initializing {arch} network...')
    else:
        raise NotImplementedError(f'{arch} is not an available network architecture.')

    # Initialize network
    if arch == 'resnet20':
        return resnet20(num_classes, additional_layers, resnet_variant)
    elif arch == 'resnet32':
        return resnet32(num_classes, additional_layers, resnet_variant)
    elif arch == 'resnet44':
        return resnet44(num_classes, additional_layers, resnet_variant)
    elif arch == 'resnet56':
        return resnet56(num_classes, additional_layers, resnet_variant)
    elif arch == 'resnet110':
        return resnet110(num_classes, additional_layers, resnet_variant)
    elif arch == 'vgg':
        return vgg(num_classes, additional_layers)
    elif arch == 'alexnet':
        return alexnet(num_classes, additional_layers)
    elif arch == 'lenet':
        return lenet(num_classes, additional_layers)


    








