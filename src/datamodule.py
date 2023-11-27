import pytorch_lightning as pl

class CLIPClassDataModule(pl.LightningDataModule):
    "Datamodule class."
    def __init__(
        data_path = 'data/stairs_dataset_20231124',
    ):
        pass