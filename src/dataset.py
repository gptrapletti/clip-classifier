import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class CCDataset(Dataset):
    '''Dataset class for CLIP Classifier.'''
    def __init__(self, filepaths, gts, captions, transforms=None):
        self.filepaths = filepaths
        self.gts = gts
        self.captions = captions
        self.transforms = transforms
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = cv2.imread(self.filepaths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.from_gt_to_ohe(self.gts[idx])
        caption = self.shorten_caption_if_needed(
            caption=self.captions[idx],
            processor=self.processor,
            max_tokens=70
        )

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label, caption
        
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
    
    def count_tokens_by_processor(self, processor, text):
        return len(processor(text=text)['input_ids'])
    
    def shorten_caption_if_needed(self, caption, processor, max_tokens):
        '''To ensure the caption is not too long (CLIP has max number of tokens)'''
        while self.count_tokens_by_processor(processor, caption) > max_tokens:
            words = caption.split()[:-1]
            caption = ' '.join(words)
        
        return caption
       
