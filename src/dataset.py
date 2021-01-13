import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

def get_datamodule(args):
    if args.dataset == 'cifar10':
        print(f'Loading {args.dataset.upper()} dataset...')
        return CIFAR10DataModule(args.batch_size, args.data_path, args.workers)
    elif args.dataset == 'cifar100':
        print(f'Loading {args.dataset.upper()} dataset...')
        return CIFAR100DataModule(args.batch_size, args.data_path, args.workers)
    elif args.dataset == 'celeba':
        print(f'Loading {args.dataset.upper()} dataset...')
        raise NotImplementedError('{} is not implemented yet'.format(args.dataset))
    elif args.dataset == 'cub200':
        print(f'Loading {args.dataset.upper()} dataset...')
        raise NotImplementedError('{} is not implemented yet'.format(args.dataset))
    else:
        raise NotImplementedError('{} is not an available dataset'.format(args.dataset))


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, workers):
        super(CIFAR10DataModule, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.workers = workers

        self.dims = (3, 32, 32)
        self.num_classes = 10
        
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])]
        )
        
    def prepare_data(self):
        CIFAR10(self.data_path, train=True, download=True)
        CIFAR10(self.data_path, train=False, download=True)

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.cifar_train = CIFAR10(self.data_path, train=True, transform=self.train_transforms)
            self.cifar_val = CIFAR10(self.data_path, train=False, transform=self.test_transforms)
        elif stage == 'test':
            self.cifar_test = CIFAR10(self.data_path, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers, pin_memory=True)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, workers):
        super(CIFAR100DataModule, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.workers = workers

        self.dims = (3, 32, 32)
        self.num_classes = 10
        
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])]
        )
        
    def prepare_data(self):
        CIFAR100(self.data_path, train=True, download=True)
        CIFAR100(self.data_path, train=False, download=True)

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.cifar_train = CIFAR100(self.data_path, train=True, transform=self.train_transforms)
            self.cifar_val = CIFAR100(self.data_path, train=False, transform=self.test_transforms)
        elif stage == 'test':
            self.cifar_test = CIFAR100(self.data_path, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers, pin_memory=True)