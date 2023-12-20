from pytorch_lightning import Trainer

def get_trainer(n_epochs, logger, callbacks, compute, devices):
    return Trainer(
        max_epochs=n_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        gradient_clip_val=0.5,
        accelerator=compute,
        devices=devices,
        # gradient_clip_algorithm='value',
    )