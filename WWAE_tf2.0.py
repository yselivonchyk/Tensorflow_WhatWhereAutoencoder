# title           :WWAE_tf2.0.py
# description     :Implementation of What-Whaere Autoencoder with help of Tensorflow 2.0+
# author          :yselivonchyk
# date            :20190405
# modeldetails    :non-sequential model, parallel training as a multiple output model

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import time
import tensorflow.nn as nn
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class SETTINGS():
  alpha = 0.1  # Determines the weight of predicted_reconstruction error
  pool_size = 7  # Determine pooling size in MNIST experiment with reconstruction
  logdir = './log/WWAE20'
  max_epochs = 50  # Train for at most this number of epochs
  report_every = 100  # Print info every NUM batches
  batch_size = 10
  learning_rate = 0.0001


def max_pool_with_argmax(net, stride):
  """
  Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
  Therefore, we use max_pool_with_argmax to extract mask and
  plain max_pool for, eeem... max_pooling.
  """
  with tf.compat.v1.name_scope('MaxPoolArgMax'):
    _, mask = tf.nn.max_pool_with_argmax(
      net,
      ksize=[1, stride, stride, 1],
      strides=[1, stride, stride, 1],
      padding='SAME')
    mask = tf.stop_gradient(mask)
    net = tf.nn.max_pool2d(net, ksize=[stride, stride], strides=SETTINGS.pool_size, padding='SAME')
    return net, mask


# Thank you, @https://github.com/Pepslee
def unpool(net, mask, stride):
  assert mask is not None
  with tf.compat.v1.name_scope('UnPool2D'):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(input=net)
    indices = tf.transpose(a=tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


def upsample(net, stride, mode='ZEROS'):
  """
  Imitate reverse operation of Max-Pooling by either placing original max values
  into a fixed postion of upsampled cell:
  [0.9] =>[[.9, 0],   (stride=2)
           [ 0, 0]]
  or copying the value into each cell:
  [0.9] =>[[.9, .9],  (stride=2)
           [ .9, .9]]

  :param net: 4D input tensor with [batch_size, width, heights, channels] axis
  :param stride:
  :param mode: string 'ZEROS' or 'COPY' indicating which value to use for undefined cells
  :return:  4D tensor of size [batch_size, width*stride, heights*stride, channels]
  """
  assert mode in ['COPY', 'ZEROS']
  with tf.compat.v1.name_scope('Upsampling'):
    net = _upsample_along_axis(net, 2, stride, mode=mode)
    net = _upsample_along_axis(net, 1, stride, mode=mode)
    return net


def _upsample_along_axis(volume, axis, stride, mode='ZEROS'):
  shape = volume.get_shape().as_list()

  assert mode in ['COPY', 'ZEROS']
  assert 0 <= axis < len(shape)

  target_shape = shape[:]
  target_shape[axis] *= stride

  padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'ZEROS' else volume
  parts = [volume] + [padding for _ in range(stride - 1)]
  volume = tf.concat(parts, min(axis + 1, len(shape) - 1))

  volume = tf.reshape(volume, target_shape)
  return volume


def conv2d(inputs, filters, kernel):
  in_channels = list(inputs.shape)[-1]
  weight_shape = kernel + [in_channels, filters]
  filters = tf.Variable(tf.initializers.GlorotUniform()(weight_shape))
  return tf.nn.conv2d(inputs, filters, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_transpose(inputs, filters, kernel):
  in_shape = list(inputs.shape)
  in_channels = list(inputs.shape)[-1]
  weight_shape = kernel + [filters, in_channels]
  weights = tf.Variable(tf.initializers.GlorotUniform()(weight_shape))
  return tf.nn.conv2d_transpose(inputs, weights, output_shape=in_shape[:-1] + [filters], strides=[1, 1, 1, 1])


def flatten(inputs):
  return tf.reshape(inputs, [tf.shape(inputs)[0], -1])


class WhatWhereAutoencoder():
  dataset = None
  _batch_shape = None

  _step = None
  _current_step = None

  def get_epoch_size(self):
    return self.dataset.shape[0] / SETTINGS.batch_size

  def get_image_shape(self):
    return self._batch_shape[2:]

  def build_mnist_model(self, input, use_unpooling):
    """
    Build autoencoder model for mnist dataset as described in the Stacked What-Where autoencoders paper

    :param input: 4D tensor of source data of shae [batch_size, w, h, channels]
    :param use_unpooling: indicate whether unpooling layer should be used instead of naive upsampling
    :return: tuple of tensors:
      train - train operation
      encode - bottleneck tensor of the autoencoder network
      decode - reconstruction of the input
    """
    # Encoder. (16)5c-(32)3c-Xp
    net = conv2d(input, 16, [5, 5])
    net = conv2d(net, 32, [3, 3])

    if use_unpooling:
      encode, mask = max_pool_with_argmax(net, SETTINGS.pool_size)
      net = unpool(encode, mask, stride=SETTINGS.pool_size)
    else:
      encode = tf.nn.max_pool2d(
        net,
        ksize=[SETTINGS.pool_size, SETTINGS.pool_size],
        strides=SETTINGS.pool_size,
        padding='SAME'
      )
      net = upsample(encode, stride=SETTINGS.pool_size)

    # Decoder
    net = conv2d_transpose(net, 16, [3, 3])
    net = conv2d_transpose(net, 1, [5, 5])
    decode = net

    loss_l2 = tf.nn.l2_loss(flatten(input) - flatten(net))

    # Optimizer
    # tf.optimizers.Adam
    train = tf.compat.v1.train.AdamOptimizer(learning_rate=SETTINGS.learning_rate).minimize(loss_l2)
    return train, encode, decode

  def fetch_dataset(self):
    mnist = tf.keras.datasets.mnist
    (train_images, _), (_, _) = mnist.load_data()  # we only need train images here [60000, 28, 28]
    if len(train_images.shape) == 3: train_images = train_images.reshape(list(train_images.shape) + [1])
    self.dataset = train_images
    self._batch_shape = [SETTINGS.batch_size, 28, 28, 1]

  def _batch_generator(self, shuffle=True):
    """Returns BATCH_SIZE of images"""
    self.permutation = np.arange(len(self.dataset) - 2)
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)

    total_batches = int(len(self.permutation) / SETTINGS.batch_size)

    for i in range(total_batches):
      batch_indexes = self.permutation[i * SETTINGS.batch_size:(i + 1) * SETTINGS.batch_size]
      yield self.dataset[batch_indexes]

  # TRAIN

  def train(self, epochs_to_train=5):
    self.fetch_dataset()

    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.compat.v1.assign(self._current_step, self._current_step + 1)

    # build models
    input = tf.compat.v1.placeholder(tf.float32, self._batch_shape, name='input')
    train, encode, decode = self.build_mnist_model(input, use_unpooling=True)  # Autoencoder using Where information
    naive_train, naive_encode, naive_decode = self.build_mnist_model(input, use_unpooling=False)  # regular Autoencoder
    # build summary with decode images
    stitched_decodings = tf.concat((input, decode, naive_decode), axis=2)
    decoding_summary_op = tf.compat.v1.summary.image('source/whatwhere/stacked', stitched_decodings)

    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      self._register_training_start(sess)

      # MAIN LOOP
      start = time.time()
      for current_epoch in xrange(epochs_to_train):
        for batch in self._batch_generator():
          _, _, decoding_summary, step = sess.run(
            [train, naive_train, decoding_summary_op, self._step],
            feed_dict={input: batch})
          self._register_batch(batch, decoding_summary, step, time.time() - start)
        self._register_epoch(current_epoch, epochs_to_train)

  def _register_training_start(self, sess):
    self.summary_writer = tf.compat.v1.summary.FileWriter('./tmp/', sess.graph)

  def _register_batch(self, batch, decoding_summary, step, elapsed):
    if step % self.get_epoch_size() % SETTINGS.report_every == 0:
      print('\r step: %6d/%4d \tbatch_per_sec: %04.1f' % (step, self.get_epoch_size(), step / elapsed), end='')
      self.summary_writer.add_summary(decoding_summary)

  def _register_epoch(self, epoch, total_epochs):
    print(' Epoch: %2d/%2d' % (epoch + 1, total_epochs))


if __name__ == '__main__':
  model = WhatWhereAutoencoder()
  model.train(SETTINGS.max_epochs)
