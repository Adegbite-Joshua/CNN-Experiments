import torch
import lightning as L
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform=transforms.ToTensor()):
        super().__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        
    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=False)
        datasets.MNIST(self.data_dir, train=False, download=False)
        
    def setup(self, stage=None):
        entire_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            download=False,
            transform=self.transform
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000,10000])
        
        self.test_ds = datasets.MNIST(
            self.data_dir,
            train=False,
            download=False,
            transform=self.transform
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False
        )
        