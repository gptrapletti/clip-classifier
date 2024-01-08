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
        encoder_name,
        image_path,
        gt_path,
        caption_path,
        val_size,
        test_size,
        batch_size,
        seed 
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.image_path = image_path
        self.gt_path = gt_path
        self.caption_path = caption_path
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.seed = seed
        self.gt_df = pd.read_csv(gt_path)
        self.caption_df = pd.read_csv(caption_path)
        self.clipclass_transforms = CCTransforms(encoder_name=self.encoder_name)

    def prepare_data(self):
        """Prepare filepaths and GTs for train, val and test sets. Filepaths are lists of filepath strings,
        GTs are list of strings (e.g. ['angular', 'bent', 'bent', 'straight', 'angular', ...]).        
        They will be used by the `setup` function to instantiate the datasets."""
        filepaths = sorted([os.path.join(self.image_path, fp) for fp in os.listdir(self.image_path)])
        gts = self.gt_df.GT.to_list()
        captions = self.caption_df.caption.to_list()

        train_filepaths, val_filepaths, train_gts, val_gts, train_captions, val_captions = train_test_split(
            filepaths,
            gts,
            captions,
            test_size=self.val_size,
            stratify=gts,
            random_state=self.seed
        )

        self.train_filepaths, self.train_gts, self.train_captions = train_filepaths, train_gts, train_captions
        self.val_filepaths, self.val_gts, self.val_captions = val_filepaths, val_gts, val_captions
        # self.test_filepaths, self.test_gts = test_filepaths, test_gts

    def setup(self, stage):
        '''Creates datasets and dataloaders for the train, val and test phases.'''
        if stage in ['fit', 'train']:
            self.train_dataset = CCDataset(
                filepaths=self.train_filepaths, 
                gts=self.train_gts,
                captions=self.train_captions, 
                transforms=self.clipclass_transforms.train_transforms
            )
            self.val_dataset = CCDataset(
                filepaths=self.val_filepaths,
                gts=self.val_gts,
                captions=self.val_captions,
                transforms=self.clipclass_transforms.test_transforms
            )

        if stage == 'test':
            raise NotImplementedError("Predicting is not implemented yet.")

        if stage == 'predict':
            # Prediction is done on the validation set
            self.predict_filepaths = self.val_filepaths
            self.predict_dataset = CCDataset(
                filepaths=self.val_filepaths,
                gts=self.val_gts, 
                captions=self.val_captions,
                transforms=self.clipclass_transforms.test_transforms
            )

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(dataset=self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
     
if __name__ == "__main__":
    from src.datamodule import CCDataModule
    dm = CCDataModule()
    dm.prepare_data()
    dm.setup('train')
    ds = dm.train_dataset
    images, labels = ds[0]

