import torch
import torch.nn as nn
from models import ResNet50, ViT_S16
from dataset import DatasetLoader
from config import cfg

def evaluate(model, val_loader, criterion, device):
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return accuracy, avg_loss

_, val_loader = DatasetLoader(cfg.DATA_DIR, cfg.BATCH_SIZE, cfg.NUM_WORKERS).get_loaders()

if cfg.MODEL_TYPE == "ResNet50":
    model = ResNet50(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)
elif cfg.MODEL_TYPE == "ViT-S/16":
    model = ViT_S16(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)
else:
    raise ValueError("Invalid model type")

criterion = nn.CrossEntropyLoss()

evaluate(model, val_loader, criterion, cfg.DEVICE)
