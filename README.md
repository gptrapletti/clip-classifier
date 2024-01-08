# CLIP Classifier

## Introduction
This project aims at exploring the use of the CLIP model (https://arxiv.org/abs/2103.00020) on top of a classifier. CLIP's rich understanding of visual features is leveraged to help in the training of a deep learning network for image classification. Conversely, the addition of a classifier is thought to be important for learning domain-specific features that CLIP might overlook. Textual caption can also be incorporated into the pipeline to assist in this task.

## Installation
1. Create conda env: `conda create --name clipclass python=3.9`
2. Install PyTorch for DGX Nvidia driver compatibility: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. Install requirements.
4. Install package `src`.

## Dataset
The dataset comprises images of staircases, representing a specific domain that challenges the capabilities of CLIP as an encoder. It can be downloaded [here](https://drive.google.com/file/d/1R5IoYjMOVfBTw3Afz9mrEHbg_czxAYXO/view?usp=sharing) (access needed). 

## ToDO
- Val dataloader shuffle back to false!
- Finish predicting via callback (use cfg).
- Use cfg for evaluate fn. Add confusion matrix and other stats too.
- Re-train with new split (just train and val) and write results to report.
- Unfreeze some resnet layers and add to report.
- Add text data.
- Replace `instantiate_encoder` function with an encoder class.
- Add resnet 18.


#### ResNet50 performances on Caltech256 dataset, 3 classes (birds), 65 images per class.

- PRETRAINED: True
    - n unfrozen layers:
        - 25: train
        - 50: train
        - 65: train but slow and caps at metric ~ 0.70
        - 75: no train
        - 100: no train

- PRETRAINED: False
    - n unfrozen layers:
        - 25: no train
        - 50: no train
        - 75: no train
        - 100: no train


TODO: report for the different models. Try resnet + large decoder! 