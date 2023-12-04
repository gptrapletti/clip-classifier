from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def get_callbacks(ckp_path):
    callbacks = [
        ModelCheckpoint(
            dirpath=ckp_path,
            filename='{epoch}',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    return callbacks