This is project includes the codes for attention-based multi-visual features fusion for object affordance detection.

**Requirements**

Python 2.7

PyTorch 0.4.1  (may not work with 1.0 or higher)

CUDA 9.0

**Dataset**

the self-collected dataset and trained models can be downloaded from [here](https://tams.informatik.uni-hamburg.de/research/datasets/index.php).

**Feature Extraction**

1. deep visual feature extract from VGG 19

patch_vgg19.py

2. deep texture feature encode by the [texture encoding network](https://hangzhang.org/PyTorch-Encoding/experiments/texture.html). Please install required sources according to the website.

patch_texture_encoding.py

**Train**

train_val.py

**Test**

affor_multi_recognition.py


