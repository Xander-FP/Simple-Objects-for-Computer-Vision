import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from CustomDataset import CustomDataset, CustomCIFAR10, CustomDTD, CustomCropDataset
from PIL import ImageStat

class DataPrep:
    def get_datasets(self, data_dir, model):
        # TODO: Let CustomDataset handle CIFAR10 and ImageNet retrieval
        if 'enerated' in data_dir:
            train_set = CustomDataset(data_path=data_dir, model=model)
            valid_set = CustomDataset(data_path=data_dir, model=model)
        else:
            train_set = CustomCropDataset(data_path=data_dir, csv_name='Train.csv')
            valid_set = CustomCropDataset(data_path=data_dir, csv_name='Train.csv')

        return train_set, valid_set
    
    def get_test_datasets(self, data_dir):
        return CustomCropDataset(data_path=data_dir, csv_name='Test.csv')


    def get_train_valid_loader(self, train_set, valid_set, batch_size, random_seed, valid_size=0.1, shuffle=True):
        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # DATA LOADERS - Look at params
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, sampler=train_sampler)

        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=batch_size, sampler=valid_sampler)

        return (train_loader, valid_loader)
    
    def get_test_loader(self, transform, data_dir, batch_size, shuffle=True):
        # custom_dataset = CustomDataset(data_path='path_to_your_data', transform=your_transformations)
        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        # data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        return data_loader
    
    def compute_mean_std(self, data_set, normalize = True):
        rgb_sum = np.array([0, 0, 0], dtype=np.float64)
        rgb_sum_sqrd = np.array([0, 0, 0], dtype=np.float64)
        count = 0
        for image, label in data_set:
            if image == None:
                break
            stat = ImageStat.Stat(image)
            rgb_sum += np.array(stat.sum)
            rgb_sum_sqrd += np.array(stat.sum2)
            count += stat.count[0]
        rgb_mean = rgb_sum / count
        rgb_stddev = np.sqrt(rgb_sum_sqrd/count - rgb_mean**2)
        if normalize:
            rgb_mean = rgb_mean/256
            rgb_stddev = rgb_stddev/256
        return {'mean': rgb_mean.tolist(), 'std': rgb_stddev.tolist()}