import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CCDataset(Dataset):
    '''Dataset class for CLIP Classifier.'''
    def __init__(self, filepaths, gts, transforms=None):
        self.filepaths = filepaths
        self.gts = gts
        self.transforms = transforms

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = cv2.imread(self.filepaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.from_gt_to_ohe(self.gts[idx])

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label
        
    def from_gt_to_ohe(self, gt):
        '''Turns a GT from a string into a one-hot encoded tensor.
        'angular' --> [1, 0, 0]
        'bent' --> [0, 1, 0]
        'straight' --> [0, 0, 1]
        '''
        if gt == 'angular':
            label = 0
        elif gt == 'bent':
            label = 1
        elif gt == 'straight':
            label = 2
        else:
            raise ValueError(f"Invalid ground truth label: {gt}")

        ohe = F.one_hot(torch.tensor(label), num_classes=3).to(torch.float16)

        return ohe
       
