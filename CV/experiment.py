from models.resnet50 import ResNet50
from models.vit_s16 import ViT_S16
from dataset import DatasetLoader
from train import train
from evaluate import evaluate
from config import cfg
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def get_augmentation(aug_name):
    if aug_name == "GaussianBlur":
        return transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    elif aug_name == "RandomErasing":
        return transforms.RandomErasing(p=0.7, scale=(0.05, 0.15), ratio=(0.3, 3.3))
    return None

def run_experiment(model_type, pretrained, augmentation=None):
    save_folder = "w pretrain" if pretrained else "wo pretrain"
    aug_folder = augmentation if augmentation else "None"
    save_dir = os.path.join(cfg.LOG_DIR, model_type, save_folder, aug_folder)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nRunning experiment: Model={model_type}, Pretrained={save_folder}, Augmentation={aug_folder}")

    transform = get_augmentation(augmentation)
    train_loader, val_loader = DatasetLoader(
        cfg.DATA_DIR, 
        cfg.BATCH_SIZE, 
        cfg.NUM_WORKERS
    ).get_loaders(transform)

    # Initialize model
    if model_type == "ResNet50":
        model = ResNet50(num_classes=cfg.NUM_CLASSES, pretrained=pretrained)
    elif model_type == "ViT-S16":
        model = ViT_S16(num_classes=cfg.NUM_CLASSES, pretrained=pretrained)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.EPOCHS,
        eta_min=1e-6
    )

    train(model, train_loader, criterion, optimizer, cfg.DEVICE, 
          epochs, save_dir, model_type, pretrained, augmentation)

    accuracy, avg_loss = evaluate(model, val_loader, criterion, cfg.DEVICE)

    results = {
        "Model": model_type,
        "Pretrained": pretrained,
        "Augmentation": aug_folder,
        "Accuracy (%)": accuracy,
        "Loss": avg_loss
    }
    
    pd.DataFrame([results]).to_csv(
        os.path.join(save_dir, "training_results.csv"), 
        index=False
    )

    return results

if __name__ == "__main__":
    experiments = [
        # ResNet50 experiments
        ("ResNet50", False, None),
        # ("ResNet50", False, "GaussianBlur"),
        # ("ResNet50", False, "RandomErasing"),
        # ("ResNet50", True, None),
        # ("ResNet50", True, "GaussianBlur"),
        # ("ResNet50", True, "RandomErasing"),
        
        # ViT-S/16 experiments
        # ("ViT-S16", False, None),
        # ("ViT-S16", False, "GaussianBlur"),
        # ("ViT-S16", False, "RandomErasing"),
        # ("ViT-S16", True, None),
        # ("ViT-S16", True, "GaussianBlur"),
        # ("ViT-S16", True, "RandomErasing"),
    ]

    results = []
    for model_type, pretrained, augmentation in experiments:
        try:
            result = run_experiment(model_type, pretrained, augmentation)
            results.append(result)
            print(f"Completed experiment: {model_type} - {'w' if pretrained else 'wo'} pretrain - {augmentation or 'None'}")
        except Exception as e:
            print(f"Error in experiment {model_type} - {pretrained} - {augmentation}: {str(e)}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.LOG_DIR, "experiment_results.csv"), index=False)
