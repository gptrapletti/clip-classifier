import os
import torch
import cv2
from numpy import log
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, LearningRateFinder

class LogGradNormCallback(Callback):
    """
    Logs the gradient log norm.
    Source: https://github.com/Lightning-AI/pytorch-lightning/issues/1462
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", self.log_gradient_norm(model))

    def log_gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                # Compute norm (a scalar) of parameter tensor
                param_norm = p.grad.detach().data.norm(2)
                # Square it according to L2 norm formula, then add
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        log_grad_norm = log(total_norm + 1e-6)
        return log_grad_norm


class EvaluationCallback(Callback):
    def __init__(self, report_dir='reports'):
        super().__init__()
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.predictions = []
      
    def on_predict_epoch_start(self, trainer, pl_module):
        self.predictions = [] # clear at the start
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.predictions.extend(outputs) # output = preds from pl_module's predict_step
        
    def on_predict_epoch_end(self, trainer, pl_module, outputs):
        gts = trainer.datamodule.predict_dataset.gts
        filepaths = trainer.datamodule.predict_filepaths
        correct_predictions_pdf = PdfPages('output/correct_predictions.pdf')
        wrong_predictions_pdf = PdfPages('output/wrong_predictions.pdf')
        mapping = {0: 'angular', 1: 'bent', 2: 'straight'}
        for pred, gt_str, filepath in zip(self.predictions, gts, filepaths):
            pred_label = torch.argmax(pred, axis=0).item() # from tensor[-1.2, 2.9, 4.4] to 2
            pred_str = mapping[pred_label]
            # Load image
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Create plot
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(os.path.basename(filepath))
            plt.axis('off')
            text = f'GT: {gt_str} - PRED: {pred_str}'
            plt.figtext(0.5, 0.05, text, ha="center", fontsize=12)
            # Add plot to PDF
            if pred_str == gt_str:
                correct_predictions_pdf.savefig()
            else:
                wrong_predictions_pdf.savefig()
            plt.close()

        correct_predictions_pdf.close()
        wrong_predictions_pdf.close()          

        print('\nReport generated!')


def get_callbacks(ckp_path):
    callbacks = [
        ModelCheckpoint(
            dirpath=ckp_path,
            filename='{epoch}',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval='epoch'),
        LogGradNormCallback(),
        # LearningRateFinder(),
        EvaluationCallback()
    ]
    return callbacks