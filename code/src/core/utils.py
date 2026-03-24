 # External
from pathlib import Path
from omegaconf import OmegaConf

# Config handling
def load_config(config_path: Path):
    cfg = OmegaConf.load(config_path)

    return {
        "data": cfg["data_sets"],
        "model": cfg["model"],
        "training": cfg["training"]
    }