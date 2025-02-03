import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet50, ViT_S16
from dataset import DatasetLoader
from config import cfg
from tqdm import tqdm
import os
import json

def train(model, train_loader, criterion, optimizer, device, epochs, model_name, pretrained, augmentation):
    log_dir = os.path.join(cfg.LOG_DIR, model_name, str(pretrained), augmentation if augmentation else "None")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "training_log.json")
    log_data = []

    print(f"\nTraining {model_name} | Pretrained: {pretrained} | Augmentation: {augmentation} | Epochs: {epochs}")

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} | Aug: {augmentation}")
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Completed | Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% | Aug: {augmentation}")
        
        log_data.append({
            "epoch": epoch+1, 
            "loss": avg_loss, 
            "accuracy": accuracy
        })
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)

    print(f"\nTraining Completed for {model_name} | Pretrained: {pretrained} | Augmentation: {augmentation}. Log saved to {log_file}")

if __name__ == "__main__":
    augmentation = cfg.AUGMENTATION

    train_loader, _ = DatasetLoader(cfg.DATA_DIR, cfg.BATCH_SIZE, cfg.NUM_WORKERS, augmentation).get_loaders()

    if cfg.MODEL_TYPE == "ResNet50":
        model = ResNet50(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)
        epochs = 20
    elif cfg.MODEL_TYPE == "ViT-S/16":
        model = ViT_S16(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)
        epochs = 15
    else:
        raise ValueError("Invalid model type")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    train(model, train_loader, criterion, optimizer, cfg.DEVICE, epochs, cfg.MODEL_TYPE, cfg.PRETRAINED, augmentation)
