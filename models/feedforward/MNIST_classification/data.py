from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
import config
import torch
import utils
import csv
import os


class CustomMNISTDataset(Dataset):
    def __init__(self, data, transform=None):

        super(CustomMNISTDataset, self).__init__()

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = int(self.data[index][0])
        image = np.array(self.data[index][1:]).astype(np.uint8)

        # Create a PIL Image with the read array and convert to grayscale
        # image = Image.fromarray(image.reshape((28, 28)), mode='L')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return {
            'image': torch.tensor(image) / 255.0,
            'label': torch.tensor([1.0 if i == label else 0.0 for i in range(10)])
        }


class CustomMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        download=False,
    ):

        super(CustomMNISTDataModule, self).__init__()

        # The URL will download the resource from kaggle as archive.zip.
        # Once extracted, the archive folder will contain a mnist_test.csv
        # and a mnist_train.csv. The mnist_train.csv file contains the
        # 60,000 training examples and labels. The mnist_test.csv contains
        # 10,000 test examples and labels. Each row consists of 785 values:
        # the first value is the label (a number from 0 to 9) and the remaining
        # 784 values are the pixel values (a number from 0 to 255).
        self.resource_name = "oddrationale/mnist-in-csv"
        self.test_file = os.path.join(data_dir, "data/mnist_test.csv")
        self.train_file = os.path.join(data_dir, "data/mnist_train.csv")
        self.data_dir = data_dir
        self.download = download

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def read_csv(csv_file, skip_header=True):
        data = []
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)

            # Skip the header
            if skip_header:
                next(csv_reader)

            for row in csv_reader:
                data.append(row)

        return data

    def prepare_data(self):

        if self.download:

            # Take the parent directory of the data dir
            resource_filename = os.path.join(self.data_dir, "data")
            utils.download_kaggle(self.resource_name, resource_filename)
            # utils.extract_zip(resource_filename, self.data_dir)

    def setup(self, stage=None):

        train_val_data = self.read_csv(self.train_file)
        test_data = self.read_csv(self.test_file)

        train_val_dataset = CustomMNISTDataset(
            train_val_data,
            # transform=transforms.Compose([transforms.ToTensor()])
        )
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset, [50000, 10000]
        )

        self.test_dataset = CustomMNISTDataset(
            test_data,
            # transform=transforms.Compose([transforms.ToTensor()])
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


if __name__ == '__main__':

    datamodule = CustomMNISTDataModule(
        config.DATA_DIR,
        download=True
    )
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()

    for batch in train_loader:
        images = batch['image']
        labels = batch['label']
        print(f'Images shape: {images.shape}')
        print(images[0])
        print(labels[0])
        break
