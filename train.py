import os
import yaml
import torch.cuda as cuda
from datetime import datetime
import mlflow
from src.datamodule import CCDataModule
from src.encoder import CLIPEncoder, ResNetEncoder
from src.classifier import CCClassifierSmall, CCClassifierLarge
from src.module import CCModule
from src.callbacks import get_callbacks
from src.logger import get_logger
from src.trainer import get_trainer

now = datetime.now().strftime('%Y%m%d-%H%M%S')

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['run_name'] = cfg['run_name'].replace('DATETIME', now)
cfg['ckp_path'] = cfg['ckp_path'].replace('DATETIME', now)

print('Configs')
for param, value in cfg.items():
    print(f'\t{param}: {value}')

if not os.path.exists(cfg['ckp_path']):
    os.makedirs(cfg['ckp_path'])    

print('\nInstantiating datamodule')
datamodule = CCDataModule(encoder_type=cfg['encoder'])

print('\nInstantiating encoder')
if cfg['encoder'] == 'clip':
    encoder = CLIPEncoder(version=cfg['clip_version'],  n_layers_to_unfreeze=0)
elif cfg['encoder'] == 'resnet':
    encoder = ResNetEncoder(pretrained=True, n_layers_to_unfreeze=0)

print('\nInstantiating downstream classifier')
classifier = CCClassifierSmall(encoder_type=cfg['encoder'])

print('\nInstantiating Lightning module')
module = CCModule(encoder=encoder, classifier=classifier, lr=cfg['lr'])

print('\nInstantiating trainer')
callbacks = get_callbacks(ckp_path=cfg['ckp_path'])
logger = get_logger(experiment_name=cfg['experiment_name'], run_name=cfg['run_name'])
logger.log_hyperparams(cfg)
trainer = get_trainer(
    n_epochs=cfg['epochs'], 
    logger=logger, 
    callbacks=callbacks,
    compute='gpu' if cuda.is_available() else 'cpu'
)

print('\nTraining...')
trainer.fit(model=module, datamodule=datamodule)

