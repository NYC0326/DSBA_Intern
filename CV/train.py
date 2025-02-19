import torch.nn as nn
import torch.optim as optim
from models.resnet50 import ResNet50
from models.vit_s16 import ViT_S16
from dataset import DatasetLoader
from config import cfg
from tqdm import tqdm
import os
import json
import torch.cuda.amp as amp

def train(model, train_loader, criterion, optimizer, device, epochs, save_dir, model_type, pretrained, augmentation):
    scaler = amp.GradScaler()
    log_file = os.path.join(save_dir, "training_log.json")
    log_data = []
    
    print(f"\nTraining {model_type} | Pretrained: {'w' if pretrained else 'wo'} pretrain | "
          f"Augmentation: {augmentation or 'None'} | Epochs: {epochs}")

    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{running_loss/len(train_loader):.3f}',
                'Acc': f'{acc:.2f}%'
            })

        epoch_data = {
            'epoch': epoch + 1,
            'loss': running_loss / len(train_loader),
            'accuracy': acc
        }
        log_data.append(epoch_data)
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)
            
    return log_data

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

    save_dir = os.path.join(cfg.LOG_DIR, cfg.MODEL_TYPE, str(cfg.PRETRAINED), augmentation if augmentation else "None")
    os.makedirs(save_dir, exist_ok=True)

    train(model, train_loader, criterion, optimizer, cfg.DEVICE, epochs, save_dir, cfg.MODEL_TYPE, cfg.PRETRAINED, augmentation)
