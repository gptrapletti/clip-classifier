import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from src.encoder import ResNetEncoder, CLIPEncoder

class CCModule(pl.LightningModule):
    """Module class."""
    def __init__(self, encoder, classifier, lr, scheduler_configs):
        super().__init__()
        # `save_hyperparameters` needed to simplifly load from checkpoint.
        # When you call `self.save_hyperparameters()`, it saves the arguments passed to your 
        # module's `__init__` method as hyperparameters. This is useful for keeping track of 
        # configuration and ensuring reproducibility. However, if some of these arguments are 
        # nn.Module instances (like your encoder and classifier), they don't need to be saved 
        # as hyperparameters because their state is already saved in the model's state_dict.
        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=['encoder', 'classifier'])
        self.encoder = encoder       
        self.classifier = classifier
        self.lr = lr
        self.scheduler_configs = scheduler_configs
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
        
    def predict_step(self, batch, batch_idx):
        images, gts = batch
        preds = self(images)
        return preds

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = self.instantiate_scheduler(optimizer=optim, configs=self.scheduler_configs)
        return {
            'optimizer': optim, 
            'lr_scheduler': {
                'scheduler': sched, 
                'monitor': 'val_loss'
            }
        }
    
    def instantiate_scheduler(self, optimizer, configs):
        scheduler = ReduceLROnPlateau(optimizer, **configs)
        return scheduler
    
    def evaluate(self, datamodule, gpu_id=0):
        device = f'cuda:{gpu_id}'
        datamodule.setup('train')
        dl = datamodule.val_dataloader()
        filepaths = datamodule.val_filepaths
        
        preds_ls = []
        gts_ls = []
        for i, batch in dl:
            images, gts = batch
            batch = [images.to(device), gts]
            preds = self.predict_step(batch, i)
            preds_ls.append(preds.detach())
            gts_ls.append(gts)
            
        pred_labels, gt_labels = [], []
        for preds, gts in zip(preds_ls, gts_ls):
            pred_labels += torch.argmax(preds, axis=1).numpy().tolist() # [1, 0, 2, 2, ...]
            gt_labels += torch.argmax(gts, axis=1).numpy().tolist()
            
        # Maybe add the last for-loop in the previous one?
            
        pass # TODO: add plotting on pdf
            
            
            
    
    
