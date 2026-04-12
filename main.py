"""
From here training runs are controlled
"""

# Internal
from scripts.train import main as run_training

# External
from pathlib import Path



# Small experiment
experiment_path = Path('experiments', 'test_small')
config_path = experiment_path / 'config.yaml'


# TODO: Add a large configuration here



dataset = 'small.pt'
run_training(experiment_path = experiment_path,
             config_path = config_path,
             dataset_name = dataset)