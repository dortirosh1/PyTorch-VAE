import yaml
import argparse
import numpy as np
import pandas as pd
from PyTorch_VAE.models import *
from PyTorch_VAE.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.test_tube import TestTubeLogger


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False
# print(config['exp_params']['img_size'])
model = vae_models[config['model_params']['name']](**config['model_params'],inp_size=config['exp_params']['img_size'])

# read dataframe of labels and data
full_df  = pd.read_csv("/home/mmm/Desktop/Dor/Gavia_Adom/datasets/exploded_annotations.csv")
dataset_path = config['exp_params']['data_path']
experiment = VAEXperiment(model,
                          config['exp_params'],dataset_path)

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 min_epochs=1,
                 logger=tt_logger,
                 log_every_n_steps=100,
                 # train_percent_check=1.,
                 # val_percent_check=1.,
                 num_sanity_val_steps=5,
                 # early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
