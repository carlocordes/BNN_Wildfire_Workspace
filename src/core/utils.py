 # External
from pathlib import Path
from omegaconf import OmegaConf

# Config handling
def load_config(config_path: Path):
    cfg = OmegaConf.load(config_path)

    return {
        "data": cfg["data"],
        "model": cfg["model"],
        "training": cfg["training"],
        "data_paths" : cfg["data_paths"],
        "data_sets" : cfg["data_sets"],
    }