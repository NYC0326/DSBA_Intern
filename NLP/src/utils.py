import omegaconf, os, sys
from omegaconf import OmegaConf


def load_config() -> omegaconf.DictConfig:
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
    config_name = sys.argv[1]
    config_path = os.path.join(config_dir, config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loaded Config: {config_path}")
    return OmegaConf.load(config_path)