import torch.nn as nn
from abc import ABC, abstractmethod

class CCClassifierBase(nn.Module, ABC):
    def __init__(self, encoder_type):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == 'clip':
            self.input_features = 512                                                               
        elif self.encoder_type == 'resnet':
            self.input_features = 1000
        else:
            raise ValueError('Encoder type not valid or missing.')

        self.model = self.backbone()

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def backbone(self):
        raise NotImplementedError("This method should be implemented by the subclass")

class CCClassifierLarge(CCClassifierBase):
    def __init__(self, encoder_type):
        super().__init__(encoder_type=encoder_type)

    def backbone(self):
        backbone = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=3)
        )
        return backbone
    
class CCClassifierSmall(CCClassifierBase):
    def __init__(self, encoder_type):
        super().__init__(encoder_type=encoder_type)

    def backbone(self):
        backbone = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=3)
        )
        return backbone

if __name__ == '__main__':
    import torch

    # CLIP
    input = torch.rand(size=[4, 512])
    classifier = CCClassifier(encoder_type='clip')
    output = classifier(input)
    print(f'Test CLIP: {output.shape}')

    # ResNet
    input = torch.rand(size=[4, 1000])
    classifier = CCClassifier(encoder_type='resnet')
    output = classifier(input)
    print(f'Test ResNet: {output.shape}')