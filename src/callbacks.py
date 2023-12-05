from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    Source: https://github.com/Lightning-AI/pytorch-lightning/issues/1462
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", self.gradient_norm(model))

    def gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm


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
        GradNormCallback()
    ]
    return callbacks