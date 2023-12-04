from src.datamodule import CCDataModule
from src.encoder import CLIPEncoder, ResNetEncoder
from src.classifier import CCClassifier
from src.module import CCModule
from src.callbacks import get_callbacks
from src.logger import get_logger
from src.trainer import get_trainer

ENCODER='clip'
CLIP_VERSION='base'
N_EPOCHS=3
LR=0.001
RUN_NAME = 'default'

EXPERIMENT_NAME = 'clip-classifier'
CKP_PATH='checkpoints'
COMPUTE='cpu'

print('Instantiating datamodule')
datamodule = CCDataModule(encoder_type=ENCODER)

print('Instantiating encoder')
if ENCODER == 'clip':
    encoder = CLIPEncoder(version=CLIP_VERSION)
elif ENCODER == 'resnet':
    encoder = ResNetEncoder(pretrained=True, n_layers_to_unfreeze=0)

print('Instantiating downstream classifier')
classifier = CCClassifier(encoder_type=ENCODER)

print('Instantiating Lightning module')
module = CCModule(encoder=encoder, classifier=classifier, lr=LR)

print('Instantiating trainer')
callbacks = get_callbacks(ckp_path=CKP_PATH)
logger = get_logger(experiment_name=EXPERIMENT_NAME, run_name=RUN_NAME)
trainer = get_trainer(n_epochs=N_EPOCHS, logger=logger, callbacks=callbacks, compute=COMPUTE)

print('Training...')
trainer.fit(model=module, datamodule=datamodule)

