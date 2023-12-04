import torch.nn as nn
import pytorch_lightning as plt

class CCModule(pl.LightningModule):
    """Module class for CLIP Classifier."""

    def __init__(self):
        super().__init__(
            self,
            encoder,
            classifier,
        )
        self.encoder = encoder
        self.classifier = classifier
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = None

    def freeze_layers(self, model, n_layers_to_unfreeze):
        '''To freeze the layers of a model. The last `n_layers_to_unfreeze` layers remain unfrozen.
        'n_layers_to_unfreeze=0' means freeze all layers, 'n_layers_to_unfreeze=1' means last 1 layer is unfrozen.
        '''
        n_tot_layers = len(list(model.parameters()))
        idxs_layers_to_freeze = list(range(0, n_tot_layers - n_layers_to_unfreeze))

        for i, param in enumerate(model.parameters()):
            if i in idxs_layers_to_freeze:    
                param.requires_grad = False

            