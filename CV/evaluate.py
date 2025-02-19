import torch
import torch.nn as nn
from tqdm import tqdm
from models.resnet50 import ResNet50
from models.vit_s16 import ViT_S16
from dataset import DatasetLoader
from config import cfg
import torchvision.transforms as transforms
import torch.cuda.amp as amp

def evaluate(model, val_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        with amp.autocast():
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                current_loss = total_loss / (total / labels.size(0))
                current_acc = 100 * correct / total
                progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2f}%"})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return accuracy, avg_loss

def run_evaluation():
    normalization = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    _, val_loader = DatasetLoader(cfg.DATA_DIR, cfg.BATCH_SIZE, cfg.NUM_WORKERS, transform=normalization).get_loaders()

    model_type = cfg.MODEL_TYPE
    pretrained = cfg.PRETRAINED

    if cfg.MODEL_TYPE == "ResNet50":
        model = ResNet50(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)
    elif cfg.MODEL_TYPE == "ViT-S16":
        model = ViT_S16(num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED)
    else:
        raise ValueError("Invalid MODEL_TYPE in config")
        
    criterion = nn.CrossEntropyLoss()
    return evaluate(model, val_loader, criterion, cfg.DEVICE)

if __name__ == "__main__":
    run_evaluation()
