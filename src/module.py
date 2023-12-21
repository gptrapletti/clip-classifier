import torch.nn as nn
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy
from tqdm import tqdm
import cv2
import os

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
    
    def evaluate(self, datamodule, device):
        datamodule.prepare_data()
        datamodule.setup('train')
        dl = datamodule.val_dataloader()
        filepaths = datamodule.val_filepaths
        
        pred_labels, gt_labels = [], []
        for i, batch in enumerate(dl):
            images, gts = batch
            batch = [images.to(device), gts]
            preds = self.predict_step(batch, i)
            pred_labels += torch.argmax(preds, axis=1).detach().cpu().numpy().tolist() # [1, 0, 2, 2, ...]
            gt_labels += torch.argmax(gts, axis=1).detach().cpu().numpy().tolist()

        correct_predictions_pdf = PdfPages('output/correct_predictions.pdf')
        wrong_predictions_pdf = PdfPages('output/wrong_predictions.pdf')

        mapping = {0: 'angular', 1: 'bent', 2: 'straight'}

        for path, pred, gt in tqdm(zip(filepaths, pred_labels, gt_labels), total=len(pred_labels)):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(os.path.basename(path))
            plt.axis('off')
            text = f'GT: {mapping[gt]} - PRED: {mapping[pred]}'
            plt.figtext(0.5, 0.05, text, ha="center", fontsize=12)

            if pred == gt:
                correct_predictions_pdf.savefig()
            else:
                wrong_predictions_pdf.savefig()

            plt.close()

        correct_predictions_pdf.close()
        wrong_predictions_pdf.close()
            
            
    
    
