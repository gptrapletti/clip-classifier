import torch
import torchvision
from transformers import CLIPModel, CLIPProcessor

class ResNetEncoder(torch.nn.Module):
    def __init__(self, pretrained, n_layers_to_unfreeze):
        super().__init__()
        self.weights = 'IMAGENET1K_V2' if pretrained else None
        self.model = torchvision.models.resnet50(weights=self.weights)
        self.freeze_layers(self.model, n_layers_to_unfreeze)

    def forward(self, x):
        return self.model(x)

    def freeze_layers(self, model, n_layers_to_unfreeze):
        '''To freeze the layers of a model. The last `n_layers_to_unfreeze` layers remain unfrozen.
        'n_layers_to_unfreeze=0' means freeze all layers, 'n_layers_to_unfreeze=1' means last 1 layer is unfrozen.
        'n_layers_to_unfreeze=-1` to have all the layers unfrozen.
        '''
        n_tot_layers = len(list(model.parameters()))
        
        if n_layers_to_unfreeze == -1:
            idxs_layers_to_freeze = []
        else:
            idxs_layers_to_freeze = list(range(0, n_tot_layers - n_layers_to_unfreeze))

        for i, param in enumerate(model.parameters()):
            if i in idxs_layers_to_freeze:    
                param.requires_grad = False


class CLIPEncoder(torch.nn.Module):
    def __init__(self, version, n_layers_to_unfreeze):
        super().__init__()
        self.version = self.get_version(version)
        self.model = CLIPModel.from_pretrained(self.version)
        self.freeze_layers(self.model, n_layers_to_unfreeze)

    def forward(self, x):
        return self.model.get_image_features(x)

    def get_version(self, version):
        if version == 'clip_base':
            return 'openai/clip-vit-base-patch32'
        elif version == 'clip_large':
            return 'openai/clip-vit-large-patch14'
        else:
            raise ValueError('Version not found. Version should be either "base" or "large".')

    def freeze_layers(self, model, n_layers_to_unfreeze):
        '''To freeze the layers of a model. The last `n_layers_to_unfreeze` layers remain unfrozen.
        'n_layers_to_unfreeze=0' means freeze all layers, 'n_layers_to_unfreeze=1' means last 1 layer is unfrozen.
        '''
        n_tot_layers = len(list(model.parameters()))
        idxs_layers_to_freeze = list(range(0, n_tot_layers - n_layers_to_unfreeze))

        for i, param in enumerate(model.parameters()):
            if i in idxs_layers_to_freeze:    
                param.requires_grad = False 


if __name__ == '__main__':
    from src.transforms import CCTransforms
    import numpy as np

    input = np.random.randint(0, 255, (512, 512, 3)).astype(np.uint8) # dummy image [H, W, C]

    print('Test CLIP')
    mytransforms = CCTransforms(encoder_type='clip')
    transformed = mytransforms.train_transforms(image=input)['image'] # [C, H, W]
    transformed_batch = transformed[None, ...] # [B, C, H, W]
    model = CLIPEncoder(version='base')
    model = model.eval()
    outputs = model(transformed_batch)
    print(outputs.shape)

    print('Test ResNet')
    mytransforms = CCTransforms(encoder_type='resnet')
    transformed = mytransforms.train_transforms(image=input)['image'] # [C, H, W]
    transformed_batch = transformed[None, ...] # [B, C, H, W]
    model = ResNetEncoder(pretrained=True, n_layers_to_unfreeze=0)
    model = model.eval()
    outputs = model(transformed_batch)
    print(outputs.shape)


