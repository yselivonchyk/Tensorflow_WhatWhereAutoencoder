# What-Where autoencoder. Tensorflow

This project contains Tensorflow implementation of [Stacked What-Where Auto-encoders](https://arxiv.org/abs/1506.02351). Implementation uses transposed convolutions provided by tensorflow and custom upsampling and unpooling code.

**Note:** As of now, unpooling code is not working with Tensorflow versions 1.13 and 2.0.0-alpha (the latest versions in pip) due to a bug in scatter_nd. It has been fixed in nightly versions and should be released with any next Tensorflow package.

## Features
- [x] Unpooling layer (implemented by @Peepslee [forum](https://github.com/tensorflow/tensorflow/issues/2169))
- [x] Fixed-position upsampling with zeros ([Inverting Visual Representations with Convolutional Networks](https://arxiv.org/abs/1506.02753))
- [x] Fixed-position upsampling with element copies
- [ ] Custom network architecture interpreter as in original paper to configure network with a single line i.e. '(16)5c-(32)3c-Xp'

## Outputs
Run tensorboard for visualization:
```bash
tensorboard --logdir=./tmp/
```

![Output example](https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/docs/tensorboard.png)

Picture above shows output images for original mnist image (left), decoding of what-where autoecoder (center), decoding of convolutional autoencoder with naive upsampling (right) while using stride=7. Picture repeat the experiment of the original paper.

![Model graph](https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/docs/graph.png)


#### Dependencies
* Python 3.5
* Tensorflow 1.0 with GPU support
* Numpy


```bash
pip3 install tensorflow-gpu numpy
```

#### Running model

Running learning script:
```bash
python WWAE.py --batch_size=128 --max_epochs=2 --pool_size=7
python WWAE_keras.py --batch_size=128 --max_epochs=2 --pool_size=7
python WWAE_tf2.0.py
```
