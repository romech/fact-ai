import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, CelebA
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import tarfile
import pandas as pd
import os
import requests
import numpy as np


def get_datamodule(args):
    if args.dataset == 'cifar10':
        print(f'Loading {args.dataset.upper()} dataset...')
        return CIFAR10DataModule(args.batch_size, args.data_path, args.workers)
    elif args.dataset == 'cifar100':
        print(f'Loading {args.dataset.upper()} dataset...')
        return CIFAR100DataModule(args.batch_size, args.data_path, args.workers)
    elif args.dataset == 'cifar100-super':
        print(f'Loading {args.dataset.upper()} dataset...')
        return CIFAR100DataModule(args.batch_size, args.data_path, args.workers, superclass=True)
    elif args.dataset == 'celeba':
        print(f'Loading {args.dataset.upper()} dataset...')
        #return CelebADataModule(args.batch_size, args.data_path, args.workers)
    elif args.dataset == 'cub200':
        print(f'Loading {args.dataset.upper()} dataset...')
        return CUB200DataModule(args.batch_size, args.data_path, args.workers)
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
    def __init__(self, batch_size, data_path, workers, superclass=False):
        super(CIFAR100DataModule, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.workers = workers
        self.superclass = superclass

        self.dims = (3, 32, 32)
        self.num_classes = 100
        
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
        if self.superclass:
            if stage == 'fit':
                self.cifar_train = CIFAR100Super(self.data_path, train=True, transform=self.train_transforms)
                self.cifar_val = CIFAR100Super(self.data_path, train=False, transform=self.test_transforms)
            elif stage == 'test':
                self.cifar_test = CIFAR100Super(self.data_path, train=False, transform=self.test_transforms)
        else:
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

class CUB200DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_path, workers):
        super(CUB200DataModule, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.workers = workers

        self.dims = (3, 224, 224)
        self.num_classes = 200
        
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4831, 0.4917, 0.4248], std=[0.1839, 0.1833, 0.1943]),
        ])
        self.test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4831, 0.4917, 0.4248], std=[0.1839, 0.1833, 0.1943])]
        )
        
    def prepare_data(self):
        CUB200Dataset(self.data_path, train=True, download=True)
        CUB200Dataset(self.data_path, train=False, download=True)

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.cub_train = CUB200Dataset(self.data_path, train=True, transform=self.train_transforms)
            self.cub_val = CUB200Dataset(self.data_path, train=False, transform=self.test_transforms)
        elif stage == 'test':
            self.cub_test = CUB200Dataset(self.data_path, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cub_train, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cub_val, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cub_test, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.workers, pin_memory=True)

class CUB200Dataset(Dataset):
    '''
    From: https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
    and https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    '''
    base_folder = 'CUB_200_2011/images'
    file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    filename = 'CUB_200_2011.tgz'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self.download()

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def check_integrity(self):
        try:
            self.load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def download(self):
        if self.check_integrity():
            print('Files already downloaded and verified')
            return

        # Could not get pytorch native google drive downloader working
        # so we curl the link instead
        print('Downloading data...')
        self.download_file_from_google_drive(self.file_id, os.path.join(self.root, self.filename))
        
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def download_file_from_google_drive(self, id, destination):
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = self.get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        self.save_response_content(response, destination)    

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CIFAR100Super(CIFAR100):
    '''
    From: https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py
    '''
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Super, self).__init__(root, train, transform, target_transform, download)

        # Map classes to their superclass
        superclass_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                       3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                       6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                       0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                       5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                       16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                       10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                       2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                       16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                       18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = superclass_labels[self.targets]

        # Update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

