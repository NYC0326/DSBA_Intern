import os
import sys
import omegaconf
from omegaconf import OmegaConf

def load_config(config_name: str = None) -> omegaconf.DictConfig:
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config_BERT.yaml"
    config_path = os.path.join(config_dir, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loaded Config: {config_path}")
    return OmegaConf.load(config_path)

