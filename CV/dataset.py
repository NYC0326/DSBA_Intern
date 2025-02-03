import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image


class DatasetLoader:
    def __init__(self, data_dir, batch_size, num_workers):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def get_augmentation(self, aug_name):
        if aug_name == "GaussianBlur":
            return transforms.Compose([
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
                self.base_transform
            ])
        elif aug_name == "RandomErasing":
            return transforms.Compose([
                self.base_transform,
                transforms.RandomErasing(p=0.7, scale=(0.05, 0.15), ratio=(0.3, 3.3))
            ])
        return self.base_transform

    def get_loaders(self, augmentation=None):
        train_data = np.load(os.path.join(self.data_dir, 'train_data.npy'))
        train_target = np.load(os.path.join(self.data_dir, 'train_target.npy'))
        test_data = np.load(os.path.join(self.data_dir, 'test_data.npy'))
        test_target = np.load(os.path.join(self.data_dir, 'test_target.npy'))

        train_data = train_data.astype(np.float32) / 255.0
        test_data = test_data.astype(np.float32) / 255.0

        train_dataset = CustomDataset(train_data, train_target, 
                                    transform=self.get_augmentation(augmentation))
        test_dataset = CustomDataset(test_data, test_target, 
                                   transform=self.base_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]
