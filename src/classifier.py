import torch.nn as nn
from abc import ABC, abstractmethod

class CCClassifierBase(nn.Module, ABC):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder_name = encoder_name

        if self.encoder_name == 'clip_base':
            self.input_features = 512
        elif self.encoder_name == 'clip_large':
            self.input_features = 768                                                               
        elif self.encoder_name in ['resnet_18', 'resnet_50']:
            self.input_features = 1000
        else:
            raise ValueError('encoder name not valid or missing.')

        self.model = self.backbone()

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def backbone(self):
        raise NotImplementedError("This method should be implemented by the subclass")

class CCClassifierLarge(CCClassifierBase):
    def __init__(self, encoder_name):
        super().__init__(encoder_name=encoder_name)

    def backbone(self):
        backbone = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=64, out_features=3)
        )
        return backbone
    
class CCClassifierSmall(CCClassifierBase):
    def __init__(self, encoder_name):
        super().__init__(encoder_name=encoder_name)

    def backbone(self):
        backbone = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=3)
        )
        return backbone

if __name__ == '__main__':
    import torch

    # CLIP
    input = torch.rand(size=[4, 512])
    classifier = CCClassifier(encoder_name='clip')
    output = classifier(input)
    print(f'Test CLIP: {output.shape}')

    # ResNet
    input = torch.rand(size=[4, 1000])
    classifier = CCClassifier(encoder_name='resnet')
    output = classifier(input)
    print(f'Test ResNet: {output.shape}')