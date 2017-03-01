from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import time
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
tf.app.flags.DEFINE_integer('pool_size', 7, 'Determine pooling size in MNIST experiment with reconstruction')
tf.app.flags.DEFINE_string('data_dir', './data/', 'MNIST dataset location')


tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')
tf.app.flags.DEFINE_integer('max_epochs', 50, 'Train for at most this number of epochs')
tf.app.flags.DEFINE_integer('report_every', 100, 'Print info every NUM batches')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')


FLAGS = tf.app.flags.FLAGS


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


def max_pool_with_argmax(net, stride):
  """
  Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
  Therefore, we use max_pool_with_argmax to extract mask and
  plain max_pool for, eeem... max_pooling.
  """
  with tf.name_scope('MaxPoolArgMax'):
    _, mask = tf.nn.max_pool_with_argmax(
      net,
      ksize=[1, stride, stride, 1],
      strides=[1, stride, stride, 1],
      padding='SAME')
    mask = tf.stop_gradient(mask)
    net = slim.max_pool2d(net, kernel_size=[stride, stride],  stride=FLAGS.pool_size)
    return net, mask


# Thank you, @https://github.com/Pepslee
def unpool(net, mask, stride):
  assert mask is not None
  with tf.name_scope('UnPool2D'):
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
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret


def upsample(net, stride):
  with tf.name_scope('Upsampling'):
    net = _upsample_along_axis(net, 2, stride, mode='ZEROS')
    net = _upsample_along_axis(net, 1, stride, mode='ZEROS')
    return net


def _upsample_along_axis(volume, axis, stride, mode='ZEROS'):
  shape = volume.get_shape().as_list()

  assert mode in ['COPY', 'ZEROS']
  assert 0 <= axis < len(shape)

  target_shape = shape[:]
  target_shape[axis] *= stride

  padding = tf.zeros(shape, dtype=volume.dtype) if mode == 'ZEROS' else volume
  parts = [volume] + [padding for _ in range(stride - 1)]
  volume = tf.concat(parts, min(axis+1, len(shape)-1))

  volume = tf.reshape(volume, target_shape)
  return volume


class WhatWhereAutoencoder():
  num_inputs = 55000

  dataset = None
  _batch_shape = None

  _step = None
  _current_step = None

  def __init__(self,
               weight_init=None,
               optimizer=tf.train.AdamOptimizer):
    self._weight_init = weight_init
    self._optimizer_constructor = optimizer

  def get_epoch_size(self):
    return self.num_inputs/FLAGS.batch_size

  def get_image_shape(self):
    return self._batch_shape[2:]

  def build_mnist_model(self, input, naive=False):
    # Encoder. (16)5c-(32)3c-Xp
    net = slim.conv2d(input, 16, [5, 5])
    net = slim.conv2d(net, 32, [3, 3])
    if not naive:
      print("UNPOOL MODE")
      print(net)
      encode, mask = max_pool_with_argmax(net, FLAGS.pool_size)
      net = unpool(encode, mask, stride=FLAGS.pool_size)
    else:
      encode = slim.max_pool2d(net, kernel_size=[FLAGS.pool_size, FLAGS.pool_size], stride=FLAGS.pool_size)
      print(encode)
      net = upsample(encode, stride=FLAGS.pool_size)

    net = slim.conv2d_transpose(net, 16, [3, 3])
    net = slim.conv2d_transpose(net, 1, [5, 5])
    decode = net

    l2rec = tf.nn.l2_loss(slim.flatten(input) - slim.flatten(net))

    # Optimizer
    optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate)
    train = optimizer.minimize(l2rec)
    return train, encode, decode

  def fetch_datasets(self):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    self.dataset = mnist.train.images.reshape((55000, 28, 28, 1))
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

  def train(self, epochs_to_train=5):
    self.fetch_datasets()

    input = tf.placeholder(tf.float32, self._batch_shape, name='input')
    self._current_step = tf.Variable(0, trainable=False, name='global_step')
    self._step = tf.assign(self._current_step, self._current_step + 1)

    train, encode, decode = self.build_mnist_model(input)
    naive_train, naive_encode, naive_decode = self.build_mnist_model(input, naive=True)
    stitched_decodings = tf.concat((input, decode, naive_decode), axis=2)
    decoding_summary_op = tf.summary.image('source/whatwhere/stacked', stitched_decodings)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      self._register_training_start(sess)

      # MAIN LOOP
      for current_epoch in xrange(epochs_to_train):
        start = time.time()
        for batch in self._batch_generator():
          _, _, decoding_summary, step = sess.run(
            [train, naive_train, decoding_summary_op, self._step],
            feed_dict={input: batch})
          self._register_batch(batch, decoding_summary, step)
        self._register_epoch(current_epoch, epochs_to_train, time.time() - start, sess)

  def _register_training_start(self, sess):
    self.summary_writer = tf.summary.FileWriter('./tmp/', sess.graph)

  def _register_batch(self, batch, decoding_summary, step):
    if step % self.get_epoch_size() % FLAGS.report_every == 0:
      print('\r step: %d/%d' % (step, self.get_epoch_size()), end='')
      self.summary_writer.add_summary(decoding_summary)

  def _register_epoch(self, epoch, total_epochs, elapsed, sess):
    print('Epoch: %d/%d' % (epoch, total_epochs))


if __name__ == '__main__':
  model = WhatWhereAutoencoder()
  model.train(FLAGS.max_epochs)
