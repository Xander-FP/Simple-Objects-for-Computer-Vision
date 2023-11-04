from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder, CIFAR10, DTD
from PIL import Image
import pandas as pd
import os

class CustomCropDataset(Dataset):
    def __init__(self, data_path, csv_name, transform=None) -> None:
        df = pd.read_csv(os.path.join(data_path, csv_name))
        # self.data = df[df['damage'] == 'DR'].reset_index(drop=True)
        self.data = df
        self.transform = transform
        if 'Test' in csv_name:
            self.data_dir = os.path.join(data_path,'test')
        else:
            self.data_dir = os.path.join(data_path,'train')
    
    def __len__(self) -> int:
        return len(self.data.index)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if index >= len(self.data.index):
            return None, None
        row = self.data.loc[index, :]
        image = Image.open(os.path.join(self.data_dir,row['filename']))

        if self.transform:
            image = self.transform(image)

        if 'test' in self.data_dir:
            return image, row['ID']
        return image, row['extent']

class CustomDataset(Dataset):
    def __init__(self, data_path, model, transform=None):
        self.new_order = None
        self.data_path = data_path
        self.transform = transform
        df = DatasetFolder(root=self.data_path, loader=Image.open, extensions=['.jpg', '.jpeg', '.png'])
        _, class_to_idx = df.find_classes(directory=self.data_path)

        self.data = DatasetFolder.make_dataset(directory=self.data_path,class_to_idx=class_to_idx,extensions=['.jpg', '.jpeg', '.png'])  # List of (image_path, label) pairs
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.new_order is not None:
            idx = self.new_order[idx][1]
        image_path, label = self.data[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def reorder(self, new_order):
        self.new_order = new_order

class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.new_order = None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.new_order is not None:
            index = self.new_order[index][1]
        return super().__getitem__(index)

    def reorder(self, new_order):
        self.new_order = new_order

class CustomDTD(DTD):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.new_order = None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.new_order is not None:
            index = self.new_order[index][1]
        return super().__getitem__(index)

    def reorder(self, new_order):
        self.new_order = new_order