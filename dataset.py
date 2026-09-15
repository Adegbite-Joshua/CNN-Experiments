import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms


DEFAULT_DATASET_NAME = "flower102"

DATASETS = {
    "mnist": {
        "class": datasets.MNIST,
        "display_name": "MNIST",
        "channels": 1,
        "num_classes": 10,
    },
    "fashion_mnist": {
        "class": datasets.FashionMNIST,
        "display_name": "FashionMNIST",
        "channels": 1,
        "num_classes": 10,
    },
    "cifar10": {
        "class": datasets.CIFAR10,
        "display_name": "CIFAR10",
        "channels": 3,
        "num_classes": 10,
    },
    "cifar100": {
        "class": datasets.CIFAR100,
        "display_name": "CIFAR100",
        "channels": 3,
        "num_classes": 100,
    },
    "flower102": {
        "class": datasets.Flowers102,
        "display_name": "Flowers102",
        "channels": 3,
        "num_classes": 102,
        "uses_split": True,
    },
}


def dataset_names():
    return list(DATASETS)


def dataset_info(name):
    return DATASETS[name]


class ImageClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        data_dir,
        batch_size,
        num_workers,
        transform=transforms.ToTensor(),
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.info = DATASETS[dataset_name]
        self.dataset_class = self.info["class"]

    def prepare_data(self):
        if self.info.get("uses_split"):
            self.dataset_class(self.data_dir, split="train", download=True)
            self.dataset_class(self.data_dir, split="val", download=True)
            self.dataset_class(self.data_dir, split="test", download=True)
            return

        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.info.get("uses_split"):
            self.train_ds = self.dataset_class(
                self.data_dir,
                split="train",
                download=False,
                transform=self.transform,
            )
            self.val_ds = self.dataset_class(
                self.data_dir,
                split="val",
                download=False,
                transform=self.transform,
            )
            self.test_ds = self.dataset_class(
                self.data_dir,
                split="test",
                download=False,
                transform=self.transform,
            )
            return

        entire_dataset = self.dataset_class(
            self.data_dir,
            train=True,
            download=False,
            transform=self.transform,
        )
        val_size = min(10000, len(entire_dataset) // 5)
        train_size = len(entire_dataset) - val_size
        self.train_ds, self.val_ds = random_split(entire_dataset, [train_size, val_size])

        self.test_ds = self.dataset_class(
            self.data_dir,
            train=False,
            download=False,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


MnistDataModule = ImageClassificationDataModule
