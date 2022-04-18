from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


class ImageDataModule(LightningDataModule):
    def __init__(self, batch_size, num_workers, seed):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = 42

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
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def download_data(self):
        # self.full_train_dataset = CIFAR10(
        #     "/tmp/data", train=True, download=True, transform=self.train_transform
        # )
        # self.full_test_dataset = CIFAR10(
        #     "/tmp/data", train=False, download=True, transform=self.test_transform
        # )

        self.train_dataset = CIFAR10(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        # self.val_dataset = CIFAR10(
        #     "/tmp/data", train=True, download=True, transform=self.test_transform
        # )
        self.test_dataset = CIFAR10(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )


class MNISTDataModule(ImageDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 33.3
        # 78.6
        normalize = transforms.Normalize(
            mean=33.3 / 255,
            std=78.6 / 255,
        )
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def download_data(self):
        self.train_dataset = MNIST(
            "/tmp/data", train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = MNIST(
            "/tmp/data", train=False, download=True, transform=self.test_transform
        )
