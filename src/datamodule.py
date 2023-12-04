import os
import pytorch_lightning as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.transforms import CCTransforms
from src.dataset import CCDataset

class CCDataModule(pl.LightningDataModule):
    "Datamodule class."
    def __init__(
        self,
        encoder_type,
        image_path = 'data/stairs_dataset_20231124',
        gt_path = 'data/stairs_dataset_annotation.csv',
        val_size = 0.2,
        test_size = 0.15,
        batch_size = 8,
        seed = 42 
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.image_path = image_path
        self.gt_path = gt_path
        self.val_size = 0.2
        self.test_size = 0.15
        self.batch_size = batch_size
        self.seed = seed
        self.df = pd.read_csv(gt_path)
        self.clipclass_transforms = CCTransforms(encoder_type=self.encoder_type)

    def prepare_data(self):
        """Prepare filepaths and GTs for train, val and test sets. Filepaths are lists of filepath strings,
        GTs are list of strings (e.g. ['angular', 'bent', 'bent', 'straight', 'angular', ...]).        
        They will be used by the `setup` function to instantiate the datasets."""
        filepaths = [os.path.join(self.image_path, fp) for fp in os.listdir(self.image_path)]
        gts = self.df.GT.to_list()

        train_val_filepaths, test_filepaths, train_val_gts, test_gts = train_test_split(
            filepaths,
            gts,
            test_size=self.test_size,
            stratify=gts,
            random_state=self.seed
        )
        train_filepaths, val_filepaths, train_gts, val_gts = train_test_split(
            train_val_filepaths,
            train_val_gts,
            test_size=self.val_size,
            stratify=train_val_gts,
            random_state=self.seed
        )

        self.train_filepaths, self.train_gts = train_filepaths, train_gts
        self.val_filepaths, self.val_gts = val_filepaths, val_gts
        self.test_filepaths, self.test_gts = test_filepaths, test_gts

    def setup(self, stage):
        '''Creates datasets and dataloaders for the train, val and test phases.'''
        if stage in ['fit', 'train']:
            self.train_dataset = CCDataset(
                filepaths=self.train_filepaths, 
                gts=self.train_gts, 
                transforms=self.clipclass_transforms.train_transforms
            )
            self.val_dataset = CCDataset(
                filepaths=self.val_filepaths,
                gts=self.val_gts, 
                transforms=self.clipclass_transforms.test_transforms
            )

        if stage == 'test':
            self.test_dataset = CCDataset(
                filepaths=self.test_filepaths, 
                gts=self.test_gts, 
                transforms=self.clipclass_transforms.test_transforms
            )

        if stage == 'predict':
            raise NotImplementedError("Predicting is not implemented yet.")

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
     
if __name__ == "__main__":
    from src.datamodule import CCDataModule
    dm = CCDataModule()
    dm.prepare_data()
    dm.setup('train')
    ds = dm.train_dataset
    images, labels = ds[0]

