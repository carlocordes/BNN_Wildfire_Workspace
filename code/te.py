"""
Random off-topic trial error script
"""

from omegaconf import OmegaConf
from pathlib import Path

cfg = OmegaConf.load("configs/project.yaml")

path = cfg['data_sets']['path']

print(Path(path))