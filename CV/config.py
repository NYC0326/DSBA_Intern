import torch
import os
import yaml

def load_yaml_config(yaml_path="configs.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
    
yaml_config = load_yaml_config()

class Config:
    DATA_DIR = os.path.join(os.getcwd(), "data")
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    EPOCHS = yaml_config["training"].get("epochs", 20)
    LEARNING_RATE = float(yaml_config["training"].get("learning_rate", 0.001))
    WEIGHT_DECAY = float(yaml_config["training"].get("weight_decay", 1e-4))
    
    SAVE_DIR = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 10
    
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "training.log")
    
    MODEL_TYPE = yaml_config.get("model_type", "ResNet50")
    PRETRAINED = yaml_config.get("pretrained", False)

cfg = Config()
