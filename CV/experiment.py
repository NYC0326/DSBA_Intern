from models import ResNet50, ViT_S16
from dataset import DatasetLoader
from train import train
from evaluate import evaluate
from config import cfg
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def get_augmentation(aug_name):
    if aug_name == "GaussianBlur":
        return transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    elif aug_name == "ColorJitter":
        return transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    else:
        return None

def run_experiment(model_type, pretrained, augmentation=None):
    print(f"\nRunning experiment: {model_type} | Pretrained: {pretrained} | Augmentation: {augmentation}")

    transform = get_augmentation(augmentation)
    train_loader, val_loader = DatasetLoader(cfg.DATA_DIR, cfg.BATCH_SIZE, cfg.NUM_WORKERS).get_loaders(transform)

    if model_type == "ResNet50":
        model = ResNet50(num_classes=cfg.NUM_CLASSES, pretrained=pretrained)
    elif model_type == "ViT-S/16":
        model = ViT_S16(num_classes=cfg.NUM_CLASSES, pretrained=pretrained)
    else:
        raise ValueError("Invalid model type")

    epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    print(f"Starting training for {model_type} | Pretrained: {pretrained} | Augmentation: {augmentation}")
    train(model, train_loader, criterion, optimizer, cfg.DEVICE, epochs, model_type, pretrained, augmentation)
    print(f"Training complete for {model_type} | Pretrained: {pretrained} | Augmentation: {augmentation}. Starting evaluation...")

    accuracy, avg_loss = evaluate(model, val_loader, criterion, cfg.DEVICE)
    print(f"Evaluation complete for {model_type} | Pretrained: {pretrained} | Augmentation: {augmentation}")

    aug_folder = augmentation if augmentation else "None"
    model_log_dir = os.path.join(cfg.LOG_DIR, model_type, str(pretrained), aug_folder)
    os.makedirs(model_log_dir, exist_ok=True)

    result_file = os.path.join(model_log_dir, "training_results.csv")
    pd.DataFrame([[model_type, pretrained, augmentation, accuracy, avg_loss]], 
                 columns=["Model", "Pretrained", "Augmentation", "Accuracy (%)", "Loss"]).to_csv(result_file, index=False)
    
    return model_type, pretrained, augmentation, accuracy, avg_loss

def save_results(results):
    df = pd.DataFrame(results, columns=["Model", "Pretrained", "Augmentation", "Accuracy (%)", "Loss"])
    df.to_csv(os.path.join(cfg.LOG_DIR, "experiment_results.csv"), index=False)
    print("Overall results saved to experiment_results.csv")

    plt.figure(figsize=(10,6))
    df.pivot(index="Model", columns=["Pretrained", "Augmentation"], values="Accuracy (%)").plot(kind="bar")
    plt.title("Model Accuracy Comparison with Augmentations")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Pretrained | Augmentation")
    plt.savefig(os.path.join(cfg.LOG_DIR, "model_comparison.png"))
    print("Saved model comparison chart.")

if __name__ == "__main__":
    results = []
    
    experiments = [
        #("ResNet50", False, None),
        ("ResNet50", False, "GaussianBlur"),
        #("ResNet50", True, None),
        ("ResNet50", True, "GaussianBlur"),
        #("ViT-S/16", False, None),
        ("ViT-S/16", False, "GaussianBlur"),
        #("ViT-S/16", True, None),
        ("ViT-S/16", True, "GaussianBlur"),
    ]

    for model_type, pretrained, augmentation in experiments:
        results.append(run_experiment(model_type, pretrained, augmentation))
        print('for loop end')

    save_results(results)
