# title           :WWAE.py
# description     :Implementation of What-Whaere Autoencoder with help of Tensorflow 1.12+
# author          :yselivonchyk
# date            :20190405


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import time
import tensorflow.contrib.slim as slim

import pooling

tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
tf.app.flags.DEFINE_integer('pool_size', 7, 'Determine pooling size in MNIST experiment with reconstruction')
tf.app.flags.DEFINE_string('data_dir', './data/', 'MNIST dataset location')

tf.app.flags.DEFINE_string('logdir', './log/WWAE', 'where to save logs.')
tf.app.flags.DEFINE_integer('max_epochs', 50, 'Train for at most this number of epochs')
tf.app.flags.DEFINE_integer('report_every', 100, 'Print info every NUM batches')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')

FLAGS = tf.app.flags.FLAGS


class WhatWhereAutoencoder():
  num_inputs = None

  dataset = None
  _batch_shape = None

  _step = None
  _current_step = None

  def get_epoch_size(self):
    return self.num_inputs / FLAGS.batch_size

  def get_image_shape(self):
    return self._batch_shape[2:]

  # DATA FEED


  def fetch_dataset(self):
    mnist = tf.keras.datasets.mnist
    (train_images, _), (_, _) = mnist.load_data()  # we only need train images here [60000, 28, 28]
    if len(train_images.shape) == 3: train_images = train_images.reshape(list(train_images.shape) + [1])
    self.dataset = train_images
    self.num_inputs = len(train_images)
    self._batch_shape = [FLAGS.batch_size, 28, 28, 1]

  def _batch_generator(self, shuffle=True):
    """Returns BATCH_SIZE of images"""
    dataset = self.dataset
    self.permutation = np.arange(len(dataset) - 2)
    self.permutation = self.permutation if not shuffle else np.random.permutation(self.permutation)

    total_batches = int(len(self.permutation) / FLAGS.batch_size)

    for i in range(total_batches):
      batch_indexes = self.permutation[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
      yield dataset[batch_indexes]

      # TRAIN

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
      net = slim.conv2d(input, 16, [5, 5])
      net = slim.conv2d(net, 32, [3, 3])

      if use_unpooling:
        encode, mask = pooling.max_pool_with_argmax(net, FLAGS.pool_size, FLAGS.pool_size)
        net = pooling.unpool(encode, mask, ksize=[1, FLAGS.pool_size, FLAGS.pool_size, 1])
      else:
        encode = slim.max_pool2d(net, kernel_size=[FLAGS.pool_size, FLAGS.pool_size], stride=FLAGS.pool_size)
        net = pooling.upsample(encode, stride=FLAGS.pool_size)

      # Decoder
      net = slim.conv2d_transpose(net, 16, [3, 3])
      net = slim.conv2d_transpose(net, 1, [5, 5])
      decode = net
      loss_l2 = tf.nn.l2_loss(slim.flatten(input) - slim.flatten(net))

      # Optimizer
      train = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss_l2)
      return train, encode, decode

  def train(self, epochs_to_train=5):
    self.fetch_dataset()

    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)

    # build models
    input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    train, encode, decode = self.build_mnist_model(input, use_unpooling=True)  # Autoencoder using Where information
    naive_train, naive_encode, naive_decode = self.build_mnist_model(input, use_unpooling=False)  # regular Autoencoder
    # build summary with decode images
    stitched_decodings = tf.concat((input, decode, naive_decode), axis=2)
    decoding_summary_op = tf.summary.image('source/whatwhere/stacked', stitched_decodings)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
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
    self.summary_writer = tf.summary.FileWriter('./tmp/', sess.graph)

  def _register_batch(self, batch, decoding_summary, step, elapsed):
    if step % self.get_epoch_size() % FLAGS.report_every == 0:
      batch_per_second = step / elapsed
      print('\r step: %6d/%4d \tbatch_per_sec: %04.1f' % (step, self.get_epoch_size(), batch_per_second), end='')
      self.summary_writer.add_summary(decoding_summary)

  def _register_epoch(self, epoch, total_epochs):
    print(' Epoch: %2d/%d' % (epoch + 1, total_epochs))


if __name__ == '__main__':
  model = WhatWhereAutoencoder()
  model.train(FLAGS.max_epochs)
