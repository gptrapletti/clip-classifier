# CLIP Classifier

## Introduction
This project aims at exploring the use of the CLIP model (https://arxiv.org/abs/2103.00020) on top of a classifier. CLIP's rich understanding of visual features is leveraged to help in the training of a deep learning network for image classification. Conversely, the addition of a classifier is thought to be important for learning domain-specific features that CLIP might overlook. Textual caption can also be incorporated into the pipeline to assist in this task.

## Dataset
The dataset comprises images of staircases, representing a specific domain that challenges the capabilities of CLIP as an encoder. It can be downloaded [here](https://drive.google.com/file/d/1R5IoYjMOVfBTw3Afz9mrEHbg_czxAYXO/view?usp=sharing) (access needed). 

## ToDO
- Add self.save_parameters() to module class.
- do trainer.predict() on val dataset, saving pdf in run dir.
- Re-train with new split (and after correction of GT)
- Replace `instantiate_encoder` function with an encoder class.
- Add resnet 18.
