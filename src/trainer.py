from pytorch_lightning import Trainer

def get_trainer(n_epochs, logger, callbacks, compute):
    return Trainer(
        max_epochs=n_epochs,
        accelerator=compute,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        gradient_clip_val=0.5,
        # gradient_clip_algorithm='value',
    )