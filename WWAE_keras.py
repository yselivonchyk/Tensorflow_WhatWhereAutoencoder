# title           :WWAE_keras.py
# description     :Implementation of What-Whaere Autoencoder with help of Tensorflow.keras
# author          :yselivonchyk
# date            :20190405
# modeldetails    :non-sequential model, parallel training as a multiple output model

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
import utils

import pooling

tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
tf.app.flags.DEFINE_integer('pool_size', 7, 'Determine pooling size in MNIST experiment with reconstruction')
tf.app.flags.DEFINE_string('data_dir', './data/', 'MNIST dataset location')

tf.app.flags.DEFINE_string('logdir', '../WWAEkeras', 'where to save logs.')
tf.app.flags.DEFINE_integer('max_epochs', 50, 'Train for at most this number of epochs')
tf.app.flags.DEFINE_integer('report_every', 100, 'Print info every NUM batches')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')
tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS


def ae_tower(input_img):
  conv1 = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
  conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  encode = layers.MaxPooling2D(pool_size=(FLAGS.pool_size, FLAGS.pool_size))(conv2)
  unpooling = layers.UpSampling2D((FLAGS.pool_size, FLAGS.pool_size))(encode)
  conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(unpooling)
  decode = layers.Conv2D(1, (5, 5), name='ae_decode', activation='sigmoid', padding='same')(conv3)
  return decode


def wwae_tower(input_img):
  conv1 = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
  conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  strides = (FLAGS.pool_size, FLAGS.pool_size)
  encode, mask_1 = pooling.MaxPoolingWithArgmax2D(pool_size=strides, strides=strides)(conv2)
  unpooling = pooling.MaxUnpooling2D((FLAGS.pool_size, FLAGS.pool_size))([encode, mask_1])
  conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(unpooling)
  decode = layers.Conv2D(1, (5, 5), name='wwae_decode', activation='sigmoid', padding='same')(conv3)
  return decode


# dataset
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255., -1)
x_test = np.expand_dims(x_test.astype('float32') / 255., -1)[:10, :, :, :]

# model
input_img = keras.Input(shape=(28, 28, 1))
reconstruction_ae = ae_tower(input_img)
reconstruction_wwae = wwae_tower(input_img)
model = keras.Model(inputs=input_img, outputs=[reconstruction_ae, reconstruction_wwae])

# Summaries
tensorboard = keras.callbacks.TensorBoard(
  log_dir=FLAGS.logdir,
  # histogram_freq=60000/FLAGS.batch_size,
  write_graph=True,
  # write_images=True
)
tensorboard.set_model(model)

# train
model.compile(
  loss={"ae_decode": keras.losses.mean_squared_error, "wwae_decode": keras.losses.mean_squared_error},
  optimizer=keras.optimizers.Adam(lr=FLAGS.learning_rate))
train_targets = {"ae_decode": x_train, "wwae_decode": x_train}
test_targets = {"ae_decode": x_test, "wwae_decode": x_test}

callback_obj = utils.TensorboardAEImageCallback('images', FLAGS.logdir)

model.fit(x_train, train_targets,
          epochs=FLAGS.max_epochs,
          batch_size=FLAGS.batch_size,
          shuffle=True,
          verbose=1,
          validation_data=(x_test, test_targets),
          callbacks=[tensorboard, callback_obj])
