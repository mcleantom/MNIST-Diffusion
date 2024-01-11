import os
from typing import Optional
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import pytorch_lightning as pl
import torch


__all__ = [
    'MNISTDataLoader',
    'NUM_WORKERS',
    'DEFAULT_DATA_DIR',
    'BATCH_SIZE',
]


NUM_WORKERS = int(os.cpu_count() / 2)
DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data'
BATCH_SIZE = 256 if torch.cuda.is_available() else 64                 


class MNISTDataLoader(pl.LightningDataModule):

    def __init__(
        self, 
        batch_size: int = BATCH_SIZE,
        data_dir: Path = DEFAULT_DATA_DIR,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2)
        ])
    
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform
    
    def train_dataloader(self):
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
