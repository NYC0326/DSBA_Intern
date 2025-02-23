import os
import sys
import wandb
from tqdm import tqdm

import torch
from torch.optim import Adam
import omegaconf
from omegaconf import OmegaConf
from transformers import set_seed

from utils import load_config
from model import EncoderForClassification
from data import get_dataloader

set_seed(42)

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

def train_iter(model, inputs, optimizer, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    loss = outputs['loss']
    accuracy = calculate_accuracy(outputs['logits'], inputs['label'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({'train_loss': loss.item(), 'train_accuracy': accuracy})
    return loss, accuracy

def valid_iter(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        loss = outputs['loss']
        accuracy = calculate_accuracy(outputs['logits'], inputs['label'])
    return loss, accuracy

def main(configs: omegaconf.DictConfig):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> DEVICE: {device}")
    
    # Load model
    model = EncoderForClassification(configs.model_config).to(device)
    
    # Load data
    train_loader = get_dataloader(configs.data_config, 'train')
    valid_loader = get_dataloader(configs.data_config, 'valid')
    test_loader  = get_dataloader(configs.data_config, 'test')
    
    # Set optimizer
    optimizer = Adam(model.parameters(), lr=configs.train_config.lr)

    # Set wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project='nlp_assignment', config=OmegaConf.to_container(configs))
    
    best_valid_accuracy = 0.0
    os.makedirs(configs.train_config.checkpoint_dir, exist_ok=True)
    
    # Train & validation for each epoch
    for epoch in range(configs.train_config.epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        for inputs in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            loss, accuracy = train_iter(model, inputs, optimizer, device)
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
        
        model.eval()
        valid_losses = []
        valid_accuracies = []
        for inputs in tqdm(valid_loader, desc=f"Valid Epoch {epoch+1}"):
            loss, accuracy = valid_iter(model, inputs, device)
            valid_losses.append(loss.item())
            valid_accuracies.append(accuracy)
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        avg_valid_accuracy = sum(valid_accuracies) / len(valid_accuracies)
        
        wandb.log({
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_train_accuracy': avg_train_accuracy,
            'avg_valid_loss': avg_valid_loss,
            'avg_valid_accuracy': avg_valid_accuracy
        })
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid Accuracy: {avg_valid_accuracy:.4f}")
        
        # Save Best Model
        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            checkpoint_path = os.path.join(configs.train_config.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("Best Model Saved")
    
    # Test Best Model
    print(">>> Testing Best Model")
    checkpoint_path = os.path.join(configs.train_config.checkpoint_dir, 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_losses = []
    test_accuracies = []
    for inputs in tqdm(test_loader, desc="Testing"):
        loss, accuracy = valid_iter(model, inputs, device)
        test_losses.append(loss.item())
        test_accuracies.append(accuracy)
    
    avg_test_loss = sum(test_losses) / len(test_losses)
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
    
    wandb.log({
        'test_loss': avg_test_loss,
        'test_accuracy': avg_test_accuracy
    })
    print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {avg_test_accuracy:.4f}")

if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_BERT.yaml"
    configs = load_config()
    print(f"Using Config: {config_name}")
    main(configs)