import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CCTransforms:
    def __init__(self, encoder_name):
        self.encoder_name = encoder_name
        self.normalization_params = self.get_normalization_params(self.encoder_name)
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()

    def get_normalization_params(self, encoder_name):
        if encoder_name.lower().split('_')[0] == 'resnet':
            return {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        elif encoder_name.lower().split('_')[0] == 'clip':
            # https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/clip/image_processing_clip.py#L173
            return {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
        else:
            raise ValueError("Wrong `encoder_name` argument.")

    def get_train_transforms(self):
        '''Transforms for the training phase.'''
        transforms = A.Compose([
            ## Resize and padding
            A.LongestMaxSize(max_size=224, interpolation=3, p=1.0), # ResNet requires 224x224 images (or resnet resizes them itself)
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0, mask_value=0, p=1.0),            
            ## Geometric transforms
            A.Affine(
                scale = (0.8, 1.2),
                rotate = (-360, 360),
                shear = (-20, 20),
                p = 0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),        
            ## Color transforms
            A.ColorJitter(
                brightness = 0.7,
                contrast = 0.7,
                saturation = 0.7,
                hue = 0.7,
                p = 0.5
            ),
            A.CLAHE(p=0.5),
            ## Normalization with values the encoder model was trained to.
            A.Normalize(
                mean=self.normalization_params['mean'], 
                std=self.normalization_params['std'], 
                p=1.0
            ),
            # To tensor
            ToTensorV2(transpose_mask=True, always_apply=True, p=1.0) # from [H, W, C] to [C, H, W]
        ])

        return transforms

    def get_test_transforms(self):
        '''Transforms for the test and validation phases. No geometric or
        color distortions go in here.'''
        transforms = A.Compose([
            A.LongestMaxSize(max_size=224, interpolation=3, p=1.0),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0, mask_value=0, p=1.0),
            A.Normalize(
                mean=self.normalization_params['mean'], 
                std=self.normalization_params['std'], 
                p=1.0
            ),
            # To tensor
            ToTensorV2(transpose_mask=True, always_apply=True, p=1.0) # from [H, W, C] to [C, H, W]
        ])

        return transforms

    