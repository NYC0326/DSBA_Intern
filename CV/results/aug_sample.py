import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dataset import DatasetLoader
from torchvision.transforms.functional import to_pil_image

augmentations = {
    "Original": transforms.Compose([]),
    "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
    "Vertical Flip": transforms.RandomVerticalFlip(p=1.0),
    "Rotation 30°": transforms.RandomRotation(30),
    "Color Jitter": transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    "Random Erasing": transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ]),
    "Gaussian Blur": transforms.GaussianBlur(kernel_size=3),
    "Affine Transform": transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
}

train_loader, _ = DatasetLoader(data_dir="data", batch_size=1, num_workers=0).get_loaders()
sample_image, _ = next(iter(train_loader))
sample_image = sample_image[0]

sample_image_pil = to_pil_image(sample_image)

num_augs = len(augmentations)
rows = 2
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

for ax, (aug_name, transform) in zip(axes.flat, augmentations.items()):
    if isinstance(transform, transforms.Compose):
        aug_image = transform(sample_image_pil)
    else:
        aug_image = transform(sample_image_pil)

    if isinstance(aug_image, torch.Tensor):
        aug_image = to_pil_image(aug_image)

    ax.imshow(aug_image)
    ax.set_title(aug_name, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("aug_sample.png", dpi=300)
plt.close()
