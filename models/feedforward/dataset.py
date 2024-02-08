from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import requests
import zipfile
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

        label = self.data[index][0]
        image = np.array(self.data[index][1:])

        # Create a PIL Image with the read array
        image = Image.fromarray(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label


class CustomMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        num_workers,
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
        self.resource_url = r"https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/download?datasetVersionNumber=2"
        self.test_file = os.path.join(data_dir, "archive/mnist_test.csv")
        self.train_file = os.path.join(data_dir, "archive/mnist_train.csv")
        self.data_dir = data_dir
        self.download = download

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def download_resource(url, filepath):

        # Make the request to download the file
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Write the content to the file
            with open(filepath, "wb") as file:
                file.write(response.content)
            print(f"Downloaded '{url}' to '{filepath}'.")
            return True
        else:
            print(f"Failed to download '{url}'. Status code: {response.status_code}.")
            return False

    @staticmethod
    def extract_resource(zip_path, directory):

        # Create the extraction directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Open the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract all contents to the specified directory
            zip_ref.extractall(directory)

        print(f"Extracted '{zip_path}' to '{directory}'")

    @staticmethod
    def read_csv(csv_file):
        data = []
        with open(csv_file, "r") as file:
            csv_reader = csv.reader(file)

            # Skip the header
            next(csv_reader)

            for row in csv_reader:
                # Convert values to integers and normalize pixel values
                label = int(row[0])
                pixels = np.array(
                    [int(val) / 255.0 for val in row[1:]], dtype=np.float32
                )
                data.append([label] + pixels.tolist())
        return data

    def prepare_data(self):

        if self.download:

            # Take the parent directory of the data dir
            parent_dir = os.path.dirname(self.data_dir)
            resource_filename = os.path.join(parent_dir, "MNIST.zip")
            self.download_resource(self.resource_url, resource_filename)
            self.extract_resource(resource_filename, self.data_dir)

    def setup(self, stage):

        train_val_data = self.read_csv(self.train_file)
        test_data = self.read_csv(self.test_file)

        train_val_dataset = CustomMNISTDataset(
            train_val_data, transform=transforms.Compose([transforms.ToTensor()])
        )
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset, [50000, 10000]
        )

        self.test_dataset = CustomMNISTDataset(
            test_data, transform=transforms.Compose([transforms.ToTensor()])
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
