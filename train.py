import os
import yaml
import torch
from datetime import datetime
from box import Box
from src.datamodule import CCDataModule
from src.classifier import CCClassifierSmall, CCClassifierLarge
from src.module import CCModule
from src.callbacks import get_callbacks
from src.logger import get_logger
from src.trainer import get_trainer
from src.utils import instantiate_encoder

now = datetime.now().strftime('%Y%m%d-%H%M%S')

with open('config.yaml', 'r') as f:
    cfg = Box(yaml.safe_load(f))

cfg.run_name = cfg.run_name.replace('DATETIME', now)
cfg.ckp_path = cfg.ckp_path.replace('DATETIME', now)

print('Configs')
for param, value in cfg.items():
    print(f'\t{param}: {value}')

if not os.path.exists(cfg.ckp_path):
    os.makedirs(cfg.ckp_path)
    
torch.set_float32_matmul_precision('medium')   

print('\nInstantiating datamodule')
datamodule = CCDataModule(
    encoder_name=cfg.encoder_name,
    image_path=cfg.image_path,
    gt_path=cfg.gt_path,
    caption_path=cfg.caption_path,
    val_size=cfg.val_size,
    test_size=cfg.test_size,
    batch_size=cfg.batch_size,
    seed=cfg.seed,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

print('\nInstantiating encoder')
encoder = instantiate_encoder(cfg.encoder_name, cfg.multimod, cfg.multimod_strategy, cfg.unfreeze, cfg.gpu_device)

print('\nInstantiating downstream classifier')
# classifier = CCClassifierSmall(encoder_name=cfg.encoder_name)
classifier = CCClassifierLarge(encoder_name=cfg.encoder_name)

print('\nInstantiating Lightning module')
module = CCModule(
    encoder=encoder, 
    classifier=classifier, 
    lr=cfg.lr, 
    scheduler_configs=cfg.scheduler,
)

print('\nInstantiating callbacks and logger')
callbacks = get_callbacks(ckp_path=cfg.ckp_path)
logger = get_logger(experiment_name=cfg.experiment_name, run_name=cfg.run_name)
logger.log_hyperparams(cfg)

# print('\nFind best learning rate...')
# from src.utils import tune_lr
# trainer = get_trainer(
#     n_epochs=cfg.epochs, 
#     logger=logger, 
#     callbacks=callbacks,
#     compute='gpu' if cuda.is_available() else 'cpu'
# )
# tune_lr(trainer=trainer, model=module, datamodule=datamodule)

print('\nTraining...')
trainer = get_trainer(
    n_epochs=cfg.epochs, 
    logger=logger, 
    callbacks=callbacks,
    gpu_device=cfg.gpu_device,
)
trainer.fit(model=module, datamodule=datamodule)



