import albumentations as A

class CLIPClassTransforms:
    def __init__(self):
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()

    def get_train_transforms(self):
        '''Transforms for the training phase.'''
        transforms = A.Compose([
            ## Resize and padding
            A.LongestMaxSize(max_size=224, interpolation=3, p=1.0),
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
            ## Normalization with values the ResNet was trained to.
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0) 
        ])

        return transforms

    def get_test_transforms(self):
        '''Transforms for the test and validation phases. No geometric or
        color distortions go in here.'''
        transforms = A.Compose([
            A.LongestMaxSize(max_size=224, interpolation=3, p=1.0),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0, mask_value=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0) 
        ])

        return transforms

    