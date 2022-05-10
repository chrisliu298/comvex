from copy import copy

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, SVHN, USPS, FashionMNIST


class ImageDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def sample_dataset(self, size):
        indices = list(range(len(self.train_dataset)))
        split = int(np.floor(size * len(self.train_dataset)))
        np.random.shuffle(indices)
        train_idx = indices[:split]
        tmp_train_dataset = copy(self.train_dataset)
        self.train_dataset.data, self.train_dataset.targets = (
            [tmp_train_dataset.data[i] for i in train_idx],
            [tmp_train_dataset.targets[i] for i in train_idx],
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        ]

    def test_dataloader(self) -> DataLoader:
        return [
            DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        ]


class CIFAR10DataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        self.train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def download_data(self):
        self.train_dataset = CIFAR10(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = CIFAR10(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )


class CIFAR10GSDataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normalize = transforms.Normalize(mean=122.6 / 255, std=61.0 / 255)
        self.train_transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor(), normalize]
        )
        self.test_transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor(), normalize]
        )

    def download_data(self):
        self.train_dataset = CIFAR10(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = CIFAR10(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )


class USPSDataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normalize = transforms.Normalize(mean=63.0 / 255, std=76.2 / 255)
        self.train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def download_data(self):
        self.train_dataset = USPS(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = USPS(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )


class SVHNDataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [111.6, 113.2, 120.6]],
            std=[x / 255.0 for x in [50.5, 51.3, 50.2]],
        )
        self.train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def download_data(self):
        self.train_dataset = SVHN(
            "/tmp/data", split="train", download=True, transform=self.train_transform
        )
        self.test_dataset = SVHN(
            "/tmp/data", split="test", download=True, transform=self.test_transform
        )

    def sample_dataset(self, size):
        indices = list(range(len(self.train_dataset)))
        split = int(np.floor(size * len(self.train_dataset)))
        np.random.shuffle(indices)
        train_idx = indices[:split]
        tmp_train_dataset = copy(self.train_dataset)
        self.train_dataset.data, self.train_dataset.labels = (
            [tmp_train_dataset.data[i] for i in train_idx],
            [tmp_train_dataset.labels[i] for i in train_idx],
        )


class MNISTDataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 33.3
        # 78.6
        normalize = transforms.Normalize(mean=33.3 / 255, std=78.6 / 255)
        self.train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def download_data(self):
        self.train_dataset = MNIST(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = MNIST(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )


class FashionMNISTDataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        normalize = transforms.Normalize(mean=72.9 / 255, std=90.0 / 255)
        self.train_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def download_data(self):
        self.train_dataset = FashionMNIST(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = FashionMNIST(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )
