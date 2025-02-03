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

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def load_data(self, file_name):
        data = np.load(os.path.join(self.data_dir, file_name))

        if data.ndim == 4 and data.shape[-1] == 3:
            data = np.transpose(data, (0, 3, 1, 2))

        data = data / 255.0
        data = (data - self.mean[:, None, None]) / self.std[:, None, None]

        return data

    def get_loaders(self, augmentation=None):
        train_data = self.load_data('train_data.npy')
        train_target = np.load(os.path.join(self.data_dir, 'train_target.npy'))
        test_data = self.load_data('test_data.npy')
        test_target = np.load(os.path.join(self.data_dir, 'test_target.npy'))

        print(f"Train Data Shape: {train_data.shape}")
        print(f"Test Data Shape: {test_data.shape}")

        transform_list = []
        
        if augmentation == "GaussianBlur":
            transform_list.append(transforms.GaussianBlur(kernel_size=3))
        elif augmentation == "RandomErasing":
            transform_list.append(transforms.RandomErasing(p=0.7, scale=(0.05, 0.15), ratio=(0.3, 3.3), value=0))

        transform = transforms.Compose(transform_list)

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data, targets, transform=None):
                self.data = data
                self.targets = targets
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                img, label = self.data[idx], self.targets[idx]
                img = np.transpose(img, (1, 2, 0))
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)

                if self.transform:
                    img = self.transform(img)

                img = transforms.ToTensor()(img)
                return img, torch.tensor(label, dtype=torch.long)

        train_dataset = CustomDataset(train_data, train_target, transform)
        test_dataset = CustomDataset(test_data, test_target, transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, test_loader
