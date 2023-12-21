# from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.exceptions import _TunerExitException

from src.encoder import CLIPEncoder, ResNetEncoder

def instantiate_encoder(encoder_name, n_layers_to_unfreeze):
    '''Takes the encoder name and number of layers to unfreeze from the end and returns
    the encoder.'''
    if encoder_name == 'clip_base':
        encoder = CLIPEncoder(version=encoder_name, n_layers_to_unfreeze=n_layers_to_unfreeze)
    elif encoder_name == 'clip_large':
        encoder = CLIPEncoder(version=encoder_name, n_layers_to_unfreeze=n_layers_to_unfreeze)
    elif encoder_name == 'resnet_50':
        encoder = ResNetEncoder(pretrained=True, n_layers_to_unfreeze=n_layers_to_unfreeze)

    return encoder

# def tune_lr(trainer, model, datamodule, iter=100):
#     '''Finds the best initial learning rate and sets it in the LightningModule object.'''
#     tuner = Tuner(trainer)

#     try:
#         tuner.lr_find(model=model, datamodule=datamodule, num_training=iter)
#     except _TunerExitException:
#         pass