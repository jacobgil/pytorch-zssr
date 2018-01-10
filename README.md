# Unofficial PyTorch implementation of  "Zero-Shot" Super-Resolution using Deep Internal Learning

Unofficial Implementation of *1712.06087 "Zero-Shot" Super-Resolution using Deep Internal Learning by Assaf Shocher, Nadav Cohen, Michal Irani.*
 
Official Project page: http://www.wisdom.weizmann.ac.il/~vision/zssr/

Paper: https://arxiv.org/abs/1712.06087


----------


This trains a deep neural network to perform super resolution using a single image.

The network is not trained on additional images, and only uses information from within the target image.
Pairs of high resolution and low resolution patches are sampled from the image, and the network fits their difference.

![Low resolution](https://github.com/jacobgil/pytorch-zssr/blob/master/examples/kennedy.png?raw=true)
![ZSSR](https://github.com/jacobgil/pytorch-zssr/blob/master/examples/kennedy_zssr.png?raw=true)

![ZSSR](https://github.com/jacobgil/pytorch-zssr/blob/master/examples/lincoln.png?raw=true)
![ZSSR](https://github.com/jacobgil/pytorch-zssr/blob/master/examples/lincoln_zssr.png?raw=true)


----------


TODO:
- Implement additional augmentation using the "Geometric self ensemble" mentioned in the paper.
- Implement gradual increase of the super resolution factor as described in the paper.
- Support for arbitrary kernel estimation and sampling with arbitrary kernels.  The current implementation interpolates the images bicubic interpolation.

Deviations from paper:
- Instead of fitting  the loss and analyzing it's standard deviation, the network is trained for a constant number of batches. The learning rate shrinks x10 every 10,000 iterations.


# Usage 
Example: ```python train.py --img img.png```
```
usage: train.py [-h] [--num_batches NUM_BATCHES] [--crop CROP] [--lr LR]
                [--factor FACTOR] [--img IMG]

optional arguments:
  -h, --help            show this help message and exit
  --num_batches NUM_BATCHES
                        Number of batches to run
  --crop CROP           Random crop size
  --lr LR               Base learning rate for Adam
  --factor FACTOR       Interpolation factor.
  --img IMG             Path to input img
```
