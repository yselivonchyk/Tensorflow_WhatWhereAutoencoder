#What-Where autoencoder. Tensorflow

This project contains Tensorflow implementation of [Stacked What-Where Auto-encoders](https://arxiv.org/abs/1506.02351). Implementation uses transposed convolutions provided by tensorflow and custom upsampling and unpooling code.

## Features
- [x] Unpooling layer (implemented by @Peepslee [forum](https://github.com/tensorflow/tensorflow/issues/2169))
- [x] Fixed-position upsampling with zeros ([Inverting Visual Representations with Convolutional Networks](https://arxiv.org/abs/1506.02753))
- [x] Fixed-position upsampling with element copies
- [ ] Custom network architecture interpreter as in original paper to configure network with a single line i.e. '(16)5c-(32)3c-Xp'

![Output example](https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/docs/tensorboard.png)
Picture above shows output images for original mnist image (left), decoding of what-where autoecoder (center), decoding of convolutional autoencoder with naive upsampling (right) while using stride=7. Picture repeat the experiment of the original paper.

####Dependencies
* Python 3.5
* Tensorflow 1.0 with GPU support
* Numpy


```bash
pip3 install tensorflow-gpu numpy
```
