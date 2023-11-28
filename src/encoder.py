import torch
import torchvision

class ResNetEncoder(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.weights = 'IMAGENET1K_V2' if pretrained else None
        self.model = torchvision.models.resnet50(weights=self.weights)

    def forward(self, x):
        return self.model(x)


class CLIPEncoder(torch.nn.Module):
    pass

if __name__ == '__main__':
    encoder = ResNetEncoder(pretrained=True)
    encoder = encoder.eval()
    input = torch.randn(1, 3, 224, 224)
    output = encoder.model(input)
    print(output.shape)

