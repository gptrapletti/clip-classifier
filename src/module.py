import torch.nn as nn
import pytorch_lightning as plt

class CLIPClassModule(pl.LightningModule):
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
