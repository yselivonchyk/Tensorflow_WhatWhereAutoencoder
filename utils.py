from PIL import Image
import tensorflow as tf
import numpy as np
import io


def merge_axis(array, source_axis=0, target_axis=1):
  array = np.moveaxis(array, source_axis, 0)
  array = np.moveaxis(array, target_axis, 1)
  array = np.concatenate(array)
  array = np.moveaxis(array, 0, target_axis - 1)
  return array


def batch_to_image_summary(batch):
  stitched = merge_axis(batch, target_axis=2)
  stitched = (255 * stitched).astype('uint8')

  stitched = stitched[:, :, 0]
  height, width = stitched.shape
  image = Image.fromarray(stitched)

  output = io.BytesIO()
  image.save(output, format='PNG')
  image_string = output.getvalue()
  output.close()
  return tf.Summary.Image(height=height,
                          width=width,
                          colorspace=1,
                          encoded_image_string=image_string)


class TensorboardAEImageCallback(tf.keras.callbacks.Callback):
  # callback visualizes input images and images of 2 outputs on a single image
  def __init__(self, tag, logdir):
    super().__init__()
    self.tag = tag
    self.logdir = logdir

  def on_epoch_end(self, epoch, logs={}):
    inp_stack = self.validation_data[0][:3]
    ae_out, wwae_out = self.model.predict(inp_stack)

    summary_str = []
    summary_str.append(tf.Summary.Value(tag=self.tag + '_input', image=batch_to_image_summary(inp_stack)))
    summary_str.append(tf.Summary.Value(tag=self.tag + '_ae', image=batch_to_image_summary(ae_out)))
    summary_str.append(tf.Summary.Value(tag=self.tag + '_wwae', image=batch_to_image_summary(wwae_out)))
    writer = tf.summary.FileWriter(self.logdir)
    writer.add_summary(tf.Summary(value=summary_str), epoch)
    return
