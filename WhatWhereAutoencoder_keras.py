# non-sequential model
# parallel training as a multiple output model
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

import pooling


tf.app.flags.DEFINE_float('alpha', 0.1, 'Determines the weight of predicted_reconstruction error')
tf.app.flags.DEFINE_integer('pool_size', 7, 'Determine pooling size in MNIST experiment with reconstruction')
tf.app.flags.DEFINE_string('data_dir', './data/', 'MNIST dataset location')

tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')
tf.app.flags.DEFINE_integer('max_epochs', 50, 'Train for at most this number of epochs')
tf.app.flags.DEFINE_integer('report_every', 100, 'Print info every NUM batches')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')
tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS


def build_ae(input_img):
    use_unpooling = False
    # Encoder. (16)5c-(32)3c-Xp
    conv1 = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    encode = layers.MaxPooling2D(pool_size=(FLAGS.pool_size, FLAGS.pool_size))(conv2)

    #decoder
    unpooling = layers.UpSampling2D((FLAGS.pool_size, FLAGS.pool_size))(encode)
    conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(unpooling)
    decode = layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(conv3)
    return decode


def build_wwae(input_img):
    use_unpooling = False
    # Encoder. (16)5c-(32)3c-Xp
    conv1 = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
    conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    encode, mask_1 = pooling.MaxPoolingWithArgmax2D(pool_size=(FLAGS.pool_size, FLAGS.pool_size))(conv2)

    #decoder
    unpooling = pooling.MaxUnpooling2D((FLAGS.pool_size, FLAGS.pool_size))([encode, mask_1])
    conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(unpooling)
    decode = layers.Conv2D(1, (5, 5), activation='sigmoid', padding='same')(conv3)
    return decode


def train(model_func):
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype('float32') / 255., -1)

    input_img = keras.Input(shape = (28, 28, 1))
    model = keras.Model(inputs=input_img, outputs=model_func(input_img))
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam())
    model.fit(x_train, x_train,
                    epochs=FLAGS.max_epochs,
                    batch_size=FLAGS.batch_size,
                    shuffle=True,
                    verbose=1
                   )

ae_model = train(build_ae)
wwae_model = train(build_wwae)
