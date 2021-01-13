from .alexnet import AlexNetEncoder, AlexNetProcessor, AlexNetDecoder
from .lenet import LeNetEncoder, LeNetProcessor, LeNetDecoder
from.vgg import VGGEncoder, VGGProcessor, VGGDecoder
from .resnet import ResNetEncoder, ResNetProcessor, ResNetDecoder

def alexnet(num_classes, additional_layers=False):
    return \
        AlexNetEncoder(additional_layers), \
        AlexNetProcessor(), \
        AlexNetDecoder(num_classes) 

def lenet(num_classes, additional_layers=False):
    return \
        LeNetEncoder(additional_layers), \
        LeNetProcessor(num_classes), \
        LeNetDecoder()
    
def vgg(num_classes, additional_layers=False):
    return \
        VGGEncoder(additional_layers), \
        VGGProcessor(), \
        VGGDecoder(num_classes)

def resnet20(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(3, additional_layers), \
        ResNetProcessor(3, variant),\
        ResNetDecoder(3, num_classes, variant)

def resnet32(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(5, additional_layers), \
        ResNetProcessor(5, variant), \
        ResNetDecoder(5, num_classes, variant)

def resnet44(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(7, additional_layers), \
        ResNetProcessor(7, variant), \
        ResNetDecoder(7, num_classes, variant)

def resnet56(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(9, additional_layers), \
        ResNetProcessor(9, variant), \
        ResNetDecoder(9, num_classes, variant)

def resnet110(num_classes, additional_layers=False, variant='alpha'):
    return \
        ResNetEncoder(18, additional_layers), \
        ResNetProcessor(18, variant), \
        ResNetDecoder(18, num_classes, variant)






