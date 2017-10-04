
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2 as inception_resnet

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_1', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_2', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_3', '', 'Path to checkpoint for inception resnet network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images

class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      _, end_points = inception.inception_v3(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    return output

class InceptionResnetModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None
    with slim.arg_scope(inception_resnet.inception_resnet_v2_arg_scope()):
      _, end_points = inception_resnet.inception_resnet_v2(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)
    self.built = True
    output = end_points['Predictions']
    return output

def predict(model,checkpoint_path):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  probs = dict()
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    predictions = model(x_input)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
          for filenames, images in load_images(FLAGS.input_dir, batch_shape):
              preds = sess.run(predictions, feed_dict={x_input: images})
              for filename, p in zip(filenames, preds):
                  probs[filename] = p

  return probs

def main(_):
  num_classes = 1001
  # Inception V3 (1)
  model = InceptionModel(num_classes)
  prob_1 = predict(model,FLAGS.checkpoint_path_1)
  # Inception V3 (2)
  model = InceptionModel(num_classes)
  prob_2 = predict(model,FLAGS.checkpoint_path_2)
  # Inception Resnet V2
  model = InceptionResnetModel(num_classes)
  prob_3 = predict(model,FLAGS.checkpoint_path_3)

  with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
    for key in prob_1.keys():
      p_ave = (prob_1[key] + prob_2[key] + prob_3[key]) / 3.
      label = np.argmax(p_ave)
      out_file.write('{0},{1}\n'.format(key, label))

if __name__ == '__main__':
  tf.app.run()
