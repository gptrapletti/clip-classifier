import torch.nn as nn
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from src.encoder import ResNetEncoder, CLIPEncoder

class CCModule(pl.LightningModule):
    """Module class."""
    def __init__(self, encoder, classifier, lr):
        super().__init__()
        self.encoder = encoder       
        self.classifier = classifier
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        self.metric = Accuracy(task='multiclass', num_classes=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, gts = batch
        preds = self(images)
        loss = self.loss_fn(input=preds, target=gts)
        metric = self.metric(
            preds=torch.argmax(preds, dim=1), # from one-hot to labels [2, 0, 1, ...]
            target=torch.argmax(gts, dim=1)
        )
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_metric', metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, gts = batch
        preds = self(images)
        loss = self.loss_fn(input=preds, target=gts)
        metric = self.metric(
            preds=torch.argmax(preds, dim=1),
            target=torch.argmax(gts, dim=1)
        )
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, gts = batch
        preds = self(images)
        metric = self.metric(
            preds=torch.argmax(preds, dim=1), 
            target=torch.argmax(gts, dim=1)
        )
        self.log('test_metric', metric, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode='min',
            factor=0.1,
            patience=10,
            threshold=0.01,
            threshold_mode='abs'
        )
        return {
            'optimizer': optim, 
            'lr_scheduler': {
                'scheduler': sched, 
                'monitor': 'val_loss'
            }
        }

            