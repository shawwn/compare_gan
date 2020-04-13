# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset loading utilities.

Creates a thin wrapper around TensorFlow Datasets (TFDS) to enable seamless
CPU/GPU/TPU workloads. The main entry point is 'get_dataset' which takes a
dataset name and a random seed and returns the corresponding tf.data.Dataset
object.

Available datasets are defined in the DATASETS dictionary. To add any dataset
supported by TFDS, simply extend the ImageDatasetV2 class as shown below with
the MNIST example and add it to DICTIONARY dictionary. Alternatively, you can
extend the ImageDatasetV2 class and load the datasets from another source.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect

from absl import flags
from absl import logging
from compare_gan.tpu import tpu_random
import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tfds_data_dir", None,
    "TFDS (TensorFlow Datasets) directory. If not set it will default to "
    "'~/tensorflow_datasets'. If the directory does not contain the requested "
    "dataset TFDS will download the dataset to this folder.")

flags.DEFINE_boolean(
    "data_fake_dataset", False,
    "If True don't load datasets from disk but create fake values.")

flags.DEFINE_integer(
    "data_shuffle_buffer_size", 10000,
    "Number of examples for the shuffle buffer.")

# Deprecated, only used for "replacing labels". TFDS will always use 64 threads.
flags.DEFINE_integer(
    "data_reading_num_threads", 64,
    "The number of threads used to read the dataset.")

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ImageNet preprocessing for ResNet."""
from absl import flags
import tensorflow as tf


IMAGE_SIZE = 512
CROP_PADDING = 32

#FLAGS = flags.FLAGS
class Namespace:
  pass

#FLAGS = Namespace()
##FLAGS.cache_decoded_image = False


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    cropped image `Tensor`
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
    if False:# FLAGS.cache_decoded_image:
      shape = tf.shape(image_bytes)
    else:
      shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    if False:#FLAGS.cache_decoded_image:
      image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                            target_height, target_width)
    else:
      image = tf.image.decode_and_crop_jpeg(
          image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  return tf.image.resize_bicubic([image], [image_size, image_size])[0]


def _decode_and_center_crop_broken(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

  return image

def _decode_and_crop_image(image_bytes, image_size, crop_padding=0, random_crop=True):
  """Crops to center of image with padding then scales image_size."""
  image = tf.io.decode_image(image_bytes, channels=3)
  shape = tf.shape(image)
  #shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]
  channels = shape[2]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  #if random_crop:
  if True:
    image = tf.image.random_crop(image, [padded_center_crop_size, padded_center_crop_size, channels])
  else:
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)
  image = tf.image.resize_area([image], [image_size, image_size])[0]
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data

def process_reals(x, drange_data = [0, 255], drange_net = [0, 1]):
  with tf.name_scope('DynamicRange'):
    x = tf.cast(x, tf.float32)
    x = adjust_dynamic_range(x, drange_data, drange_net)
  return x

def preprocess_image(image_bytes, is_training, use_bfloat16=False, image_size=IMAGE_SIZE, mirror=True, random_crop=None):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  if random_crop is None:
    random_crop = not is_training
  image = _decode_and_crop_image(image_bytes, image_size, random_crop=random_crop)
  # image = process_reals(image)
  if mirror:
    image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""


import abc
from collections import namedtuple
import functools
import math
import os
from absl import flags
import tensorflow as tf

#FLAGS = flags.FLAGS


def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    image = preprocess_image(
        image_bytes=image_bytes, is_training=False)
    return image

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label):
  """Build an Example proto for an example.

  Args:
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network

  Returns:
    Example proto
  """

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/class/label': _int64_feature(label),
              'image/encoded': _bytes_feature(image_buffer)
          }))
  return example


class ImageNetTFExampleInput(object):
  """Base class for ImageNet input_fn generator.

  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    num_cores: `int` for the number of TPU cores
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               num_cores=8,
               image_size=IMAGE_SIZE,
               prefetch_depth_auto_tune=False,
               transpose_input=False):
    self.image_preprocessing_fn = preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.num_cores = num_cores
    self.transpose_input = transpose_input
    self.image_size = image_size
    self.prefetch_depth_auto_tune = prefetch_depth_auto_tune

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      if False:#FLAGS.train_batch_size // FLAGS.num_cores > 8:
        shape = [None, None, None, batch_size]
      else:
        shape = [None, None, batch_size, None]
      images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
      images = tf.reshape(images, [-1])
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def dataset_parser(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 0

    # Return all black images for padded data.
    image = tf.cond(
        label < 0, lambda: self._get_null_input(None), lambda: self.  # pylint: disable=g-long-lambda
        image_preprocessing_fn(
            image_bytes=image_bytes,
            is_training=self.is_training,
            image_size=self.image_size,
            use_bfloat16=self.use_bfloat16))

    return image, label

  def dataset_parser_static(self, value):
    """Parses an image and its label from a serialized ResNet-50 TFExample.

       This only decodes the image, which is prepared for caching.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (image, label) from the TFExample.
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/format': tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image_bytes = tf.io.decode_jpeg(image_bytes, 3)

    # Subtract one so that labels are in [0, 1000).
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32) - 0
    return image_bytes, label

  def dataset_parser_dynamic(self, image_bytes, label):
    return self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        image_size=self.image_size,
        use_bfloat16=self.use_bfloat16), label

  def pad_dataset(self, dataset, num_hosts):
    """Pad the eval dataset so that eval can have the same batch size as training."""
    num_dataset_per_shard = int(
        math.ceil(FLAGS.num_eval_images / FLAGS.eval_batch_size) *
        FLAGS.eval_batch_size / num_hosts)
    example_string = 'dummy_string'
    padded_example = _convert_to_example(
        str.encode(example_string), -1).SerializeToString()
    padded_dataset = tf.data.Dataset.from_tensors(
        tf.constant(padded_example, dtype=tf.string))
    padded_dataset = padded_dataset.repeat(num_dataset_per_shard)

    dataset = dataset.concatenate(padded_dataset).take(num_dataset_per_shard)
    return dataset

  @abc.abstractmethod
  def make_source_dataset(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object.
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.
/
    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      if 'dataset_index' in params:
        current_host = params['dataset_index']
        num_hosts = params['dataset_num_shards']
      else:
        current_host = 0
        num_hosts = 1

    dataset = self.make_source_dataset(current_host, num_hosts)

    #if not self.is_training:
    #  # Padding for eval.
    #  dataset = self.pad_dataset(dataset, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    if self.is_training and False: #iand FLAGS.cache_decoded_image:
      dataset = dataset.apply(
          tf.contrib.data.map_and_batch(
              self.dataset_parser_dynamic,
              batch_size=batch_size,
              num_parallel_batches=self.num_cores,
              drop_remainder=True))
    else:
      dataset = dataset.apply(
          tf.contrib.data.map_and_batch(
              self.dataset_parser,
              batch_size=batch_size,
              num_parallel_batches=self.num_cores,
              drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      if FLAGS.train_batch_size // FLAGS.num_cores > 8:
        transpose_array = [1, 2, 3, 0]
      else:
        transpose_array = [1, 2, 0, 3]
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, transpose_array), labels
                                 ),
          num_parallel_calls=self.num_cores)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    if self.prefetch_depth_auto_tune:
      dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    else:
      dataset = dataset.prefetch(4)

    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    return dataset


class ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               data_dir,
               is_training=True,
               use_bfloat16=False,
               transpose_input=False,
               image_size=224,
               num_parallel_calls=64,
               num_cores=8,
               prefetch_depth_auto_tune=False,
               cache=False):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      image_size: size of input images
      num_parallel_calls: concurrency level to use when reading data from disk.
      num_cores: Number of prefetch threads
      prefetch_depth_auto_tune: Auto-tuning prefetch depths in input pipeline
      cache: if true, fill the dataset by repeating from its cache
    """
    super(ImageNetInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        num_cores=num_cores,
        prefetch_depth_auto_tune=prefetch_depth_auto_tune,
        transpose_input=transpose_input)
    self.data_dir = data_dir
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3], tf.bfloat16
                    if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    file_patterns = [x.strip() for x in self.data_dir.split(',') if len(x.strip()) > 0]

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = None
    for pattern in file_patterns:
      x = tf.data.Dataset.list_files(pattern, shuffle=False)
      if dataset is None:
        dataset = x
      else:
        dataset = dataset.concatenate(x)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      # We shuffle only during training, and during training, we must produce an
      # infinite dataset, so apply the fused shuffle_and_repeat optimized
      # dataset transformation.
      dataset = dataset.apply(
        tf.contrib.data.shuffle_and_repeat(1024 * 16))
      #dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))

    if self.is_training and False:#and FLAGS.cache_decoded_image:
      dataset = dataset.map(
          self.dataset_parser_static,
          num_parallel_calls=self.num_parallel_calls)

    if self.cache:
      dataset = dataset.cache()
    if self.is_training:
      # We shuffle only during training, and during training, we must produce an
      # infinite dataset, so apply the fused shuffle_and_repeat optimized
      # dataset transformation.
      dataset = dataset.apply(
          tf.contrib.data.shuffle_and_repeat(1024 * 16))
    return dataset


# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = namedtuple('BigtableSelection',
                               ['project',
                                'instance',
                                'table',
                                'prefix',
                                'column_family',
                                'column_qualifier'])


class ImageNetBigtableInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a Bigtable for training or evaluation.
  """

  def __init__(self, is_training, use_bfloat16, transpose_input, selection):
    """Constructs an ImageNet input from a BigtableSelection.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      selection: a BigtableSelection specifying a part of a Bigtable.
    """
    super(ImageNetBigtableInput, self).__init__(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.selection = selection

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    data = self.selection
    client = tf.contrib.cloud.BigtableClient(data.project, data.instance)
    table = client.table(data.table)
    ds = table.parallel_scan_prefix(data.prefix,
                                    columns=[(data.column_family,
                                              data.column_qualifier)])
    # The Bigtable datasets will have the shape (row_key, data)
    ds_data = ds.map(lambda index, data: data)

    if self.is_training:
      ds_data = ds_data.repeat()

    return ds_data


class ImageDatasetV2(object):
  """Interface for Image datasets based on TFDS (TensorFlow Datasets).

  This method handles both CPU/GPU and TPU data loading settings. If the flag
  --data_fake_dataset is True the methods will create a small fake dataset from
  in-memory NumPy arrays and not read from disk.
  The pipleline of input operations is as follows:
  1) Shuffle filenames (with seed).
  2) Load file content from disk. Decode images.
  Dataset content after this step is a dictionary.
  3) Prefetch call here.
  4) Filter examples (e.g. by size or label).
  5) Parse example.
  Dataset content after this step is a tuple of tensors (image, label).
  6) train_only: Repeat dataset.
  7) Transform (random cropping with seed, resizing).
  8) Preprocess (adding sampled noise/labels with seed).
  Dataset content after this step is a tuple (feature dictionary, label tensor).
  9) train only: Shuffle examples (with seed).
  10) Batch examples.
  11) Prefetch examples.

  Step 1-3 are done by _load_dataset() and wrap tfds.load().
  Step 4-11 are done by train_input_fn() and eval_input_fn().
  """

  def __init__(self,
               name,
               tfds_name,
               resolution,
               colors,
               num_classes,
               eval_test_samples,
               seed):
    logging.info("ImageDatasetV2(name=%s, tfds_name=%s, resolution=%d, "
                 "colors=%d, num_classes=%s, eval_test_samples=%s, seed=%s)",
                 name, tfds_name, resolution, colors, num_classes,
                 eval_test_samples, seed)
    self._name = name
    self._tfds_name = tfds_name
    self._resolution = resolution
    self._colors = colors
    self._num_classes = num_classes
    self._eval_test_sample = eval_test_samples
    self._seed = seed

    self._train_split = tfds.Split.TRAIN
    self._eval_split = tfds.Split.TEST

  @property
  def name(self):
    """Name of the dataset."""
    return self._name

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def eval_test_samples(self):
    """Number of examples in the "test" split of this dataset."""
    if FLAGS.data_fake_dataset:
      return 100
    return self._eval_test_sample

  @property
  def image_shape(self):
    """Returns a tuple with the image shape."""
    return (self._resolution, self._resolution, self._colors)

  def _make_fake_dataset(self, split):
    """Returns a fake data set with the correct shapes."""
    np.random.seed(self._seed)
    num_samples_per_epoch = 100
    num_epochs = self.eval_test_samples // 100 if split == "test" else None
    images_shape = [num_samples_per_epoch] + list(self.image_shape)
    images = np.random.uniform(size=images_shape).astype(np.float32)
    labels = np.ones((num_samples_per_epoch,), dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    return ds.repeat(num_epochs)

  def _get_per_host_random_seed(self, tpu_context=None):
    """Returns the dataset seed for according to the TPUContext.

    On CPU/GPU it returns the default seed. For TPUs the input_fn is executed
    on every host machine (if per-host input is set, which is set by default).
    We use a different (but deterministically computed) random seed on each host
    to ensure each host machine sees a different stream of input data.

    Args:
      tpu_context: TPU execution context.

    Returns:
      The current seed if CPU/GPU and a host-specific seed for TPU.
    """
    if self._seed is None:
      logging.warning("Dataset seed not set.")
      return None
    if tpu_context is None:
      logging.warning("No TPUContext, using unmodified dataset seed %s.",
                      self._seed)
      return self._seed
    seed = self._seed + tpu_context.current_host
    logging.info("Running with %d hosts, modifying dataset seed for "
                 "host %d to %s.", tpu_context.num_hosts,
                 tpu_context.current_host, seed)
    return seed

  @gin.configurable("replace_labels", whitelist=["file_pattern"])
  def _replace_labels(self, split, ds, file_pattern=None):
    """Replaces the labels in the dataset with labels from separate files.

    This functionality is used if one wants to either replace the labels with
    soft labels (i.e. softmax over the logits) or label the instances with
    a new classifier.

    Args:
      split: Dataset split (e.g. train/test/validation).
      ds: The underlying TFDS object.
      file_pattern: Path to the replacement files.

    Returns:
      An instance of tf.data.Dataset with the updated labels.
    """
    if not file_pattern:
      return ds
    file_pattern = file_pattern.format(split=split)
    logging.warning("Using labels from %s for split %s.", file_pattern, split)
    label_ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    label_ds = label_ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=FLAGS.data_reading_num_threads)
    ds = tf.data.Dataset.zip((ds, label_ds)).map(self._replace_label)
    return ds

  def _replace_label(self, feature_dict, new_unparsed_label):
    """Replaces the label from the feature_dict with the new label.

    Furthermore, if the feature_dict contains a key for the file_name which
    identifies an instance, we double-check that the we are replacing the label
    of the correct instance.

    Args:
      feature_dict: A serialized TFRecord containing the old label.
      new_unparsed_label: A serialized TFRecord containing the new label.

    Returns:
      Updates the label in the label dict to the new label.
    """
    label_spec = {
        "file_name": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.int64),
    }
    parsed_label = tf.parse_single_example(new_unparsed_label, label_spec)
    with tf.control_dependencies([
        tf.assert_equal(parsed_label["file_name"], feature_dict["file_name"])]):
      feature_dict["label"] = tf.identity(parsed_label["label"])
    return feature_dict

  def _parse_fn(self, features):
    image = tf.cast(features["image"], tf.float32) / 255.0
    return image, features["label"]

  def _load_dataset(self, split, params):
    """Loads the underlying dataset split from disk.

    Args:
      split: Name of the split to load.

    Returns:
      Returns a `tf.data.Dataset` object with a tuple of image and label tensor.
    """
    if FLAGS.data_fake_dataset:
      return self._make_fake_dataset(split)
    ds = tfds.load(
        self._tfds_name,
        split=split,
        data_dir=FLAGS.tfds_data_dir,
        as_dataset_kwargs={"shuffle_files": False})
    ds = self._replace_labels(split, ds)
    ds = ds.map(self._parse_fn)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)

  def _train_filter_fn(self, image, label):
    del image, label
    return True

  def _train_transform_fn(self, image, label, seed):
    del seed
    return image, label

  def _eval_transform_fn(self, image, label, seed):
    del seed
    return image, label

  def train_input_fn(self, params=None, preprocess_fn=None):
    """Input function for reading data.

    Args:
      params: Python dictionary with parameters. Must contain the key
        "batch_size". TPUEstimator will set this for you!
      preprocess_fn: Function to process single examples. This is allowed to
        have a `seed` argument.

    Returns:
      `tf.data.Dataset` with preprocessed and batched examples.
    """
    if params is None:
      params = {}
    seed = self._get_per_host_random_seed(params.get("context", None))
    logging.info("train_input_fn(): params=%s seed=%s", params, seed)

    ds = self._load_dataset(split=self._train_split, params=params)
    if hasattr(self, "_shortcircuit"):
      return ds
    ds = ds.filter(self._train_filter_fn)
    ds = ds.repeat()
    ds = ds.map(functools.partial(self._train_transform_fn, seed=seed))
    if preprocess_fn is not None:
      if "seed" in inspect.getargspec(preprocess_fn).args:
        preprocess_fn = functools.partial(preprocess_fn, seed=seed)
      ds = ds.map(preprocess_fn)
      # Add a feature for the random offset of operations in tpu_random.py.
      ds = tpu_random.add_random_offset_to_features(ds)
    ds = ds.shuffle(FLAGS.data_shuffle_buffer_size, seed=seed)
    if "batch_size" in params:
      ds = ds.batch(params["batch_size"], drop_remainder=True)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)

  def eval_input_fn(self, params=None, split=None):
    """Input function for reading data.

    Args:
      params: Python dictionary with parameters. Must contain the key
        "batch_size". TPUEstimator will set this for you!
      split: Name of the split to use. If None will use the default eval split
        of the dataset.

    Returns:
      `tf.data.Dataset` with preprocessed and batched examples.
    """
    if params is None:
      params = {}
    if split is None:
      split = self._eval_split
    seed = self._get_per_host_random_seed(params.get("context", None))
    logging.info("eval_input_fn(): params=%s seed=%s", params, seed)

    ds = self._load_dataset(split=split, params=params)
    if hasattr(self, "_shortcircuit"):
      return ds
    # No filter, no rpeat.
    ds = ds.map(functools.partial(self._eval_transform_fn, seed=seed))
    # No shuffle.
    if "batch_size" in params:
      ds = ds.batch(params["batch_size"], drop_remainder=True)
    return ds.prefetch(tf.contrib.data.AUTOTUNE)

  # For backwards compatibility ImageDataset.
  def input_fn(self, params, mode=tf.estimator.ModeKeys.TRAIN,
               preprocess_fn=None):
    assert mode == tf.estimator.ModeKeys.TRAIN, mode
    return self.train_input_fn(params=params, preprocess_fn=preprocess_fn)

  # For backwards compatibility ImageDataset.
  def load_dataset(self, split_name):
    assert split_name == "test", split_name
    return self.eval_input_fn()

class DanbooruDataset(ImageDatasetV2):

  def __init__(self, seed, resolution):
    super(DanbooruDataset, self).__init__(
        name="danbooru",
        tfds_name="danbooru",
        resolution=resolution,
        colors=3,
        num_classes=1,
        eval_test_samples=10000,
        seed=seed)
    self.resolution = resolution
    self._shortcircuit = True

  def _parse_fn(self, image, label):
    # image, label = features[0]
    # label = tf.random.uniform(shape=[], minval=0, maxval=1000, dtype=tf.int32)
    # label = tf.constant(0, dtype=tf.int32)
    # image = tf.cast(features["image"], tf.float32) / 255.0
    return image / 255.0, label

  def _load_dataset(self, split, params):
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      if 'dataset_index' in params:
        current_host = params['dataset_index']
        num_hosts = params['dataset_num_shards']
      else:
        current_host = 0
        num_hosts = 1
    num_replicas = params["context"].num_replicas if "context" in params else 1

    path = os.environ['DATASETS'] if 'DATASETS' in os.environ else "gs://danbooru-euw4a/datasets/danbooru2019-s/danbooru2019-s-0*"
    ini = ImageNetInput(
      path,
      is_training=True,
      image_size=self.resolution,
      prefetch_depth_auto_tune=True,
      num_cores=num_hosts,
    )
    iparams = dict(params)
    #iparams['batch_size'] = 1
    dset = ini.input_fn(iparams)
    dset = dset.map(self._parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print('Using dataset(s) %s (host %d / %d)' % (path, current_host, num_hosts))
    return dset

class MnistDataset(ImageDatasetV2):
  """Wrapper for the MNIST dataset from TFDS."""

  def __init__(self, seed):
    super(MnistDataset, self).__init__(
        name="mnist",
        tfds_name="mnist",
        resolution=28,
        colors=1,
        num_classes=10,
        eval_test_samples=10000,
        seed=seed)


class FashionMnistDataset(ImageDatasetV2):
  """Wrapper for the Fashion-MNIST dataset from TDFS."""

  def __init__(self, seed):
    super(FashionMnistDataset, self).__init__(
        name="fashion_mnist",
        tfds_name="fashion_mnist",
        resolution=28,
        colors=1,
        num_classes=10,
        eval_test_samples=10000,
        seed=seed)


class Cifar10Dataset(ImageDatasetV2):
  """Wrapper for the CIFAR10 dataset from TDFS."""

  def __init__(self, seed):
    super(Cifar10Dataset, self).__init__(
        name="cifar10",
        tfds_name="cifar10",
        resolution=32,
        colors=3,
        num_classes=10,
        eval_test_samples=10000,
        seed=seed)


class CelebaDataset(ImageDatasetV2):
  """Wrapper for the CelebA dataset from TFDS."""

  def __init__(self, seed):
    super(CelebaDataset, self).__init__(
        name="celeb_a",
        tfds_name="celeb_a",
        resolution=64,
        colors=3,
        num_classes=None,
        eval_test_samples=10000,
        seed=seed)

  def _parse_fn(self, features):
    """Returns 64x64x3 image and constant label."""
    image = features["image"]
    image = tf.image.resize_image_with_crop_or_pad(image, 160, 160)
    # Note: possibly consider using NumPy's imresize(image, (64, 64))
    image = tf.image.resize_images(image, [64, 64])
    image.set_shape(self.image_shape)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.constant(0, dtype=tf.int32)
    return image, label


class LsunBedroomDataset(ImageDatasetV2):
  """Wrapper from the LSUN Bedrooms dataset from TFDS."""

  def __init__(self, seed):
    super(LsunBedroomDataset, self).__init__(
        name="lsun-bedroom",
        tfds_name="lsun/bedroom",
        resolution=128,
        colors=3,
        num_classes=None,
        eval_test_samples=30000,
        seed=seed)

    # As the official LSUN validation set only contains 300 samples, which is
    # insufficient for FID computation, we're splitting off some trianing
    # samples. The smallest percentage selectable through TFDS is 1%, so we're
    # going to use that (corresponding roughly to 30000 samples).
    # If you want to use fewer eval samples, just modify eval_test_samples.
    self._train_split, self._eval_split = \
        tfds.Split.TRAIN.subsplit([99, 1])

  def _parse_fn(self, features):
    """Returns a 128x128x3 Tensor with constant label 0."""
    image = features["image"]
    image = tf.image.resize_image_with_crop_or_pad(
        image, target_height=128, target_width=128)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.constant(0, dtype=tf.int32)
    return image, label


def _transform_imagnet_image(image, target_image_shape, crop_method, seed):
  """Preprocesses ImageNet images to have a target image shape.

  Args:
    image: 3-D tensor with a single image.
    target_image_shape: List/Tuple with target image shape.
    crop_method: Method for cropping the image:
      One of: distorted, random, middle, none
    seed: Random seed, only used for `crop_method=distorted`.

  Returns:
    Image tensor with shape `target_image_shape`.
  """
  if crop_method == "distorted":
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        aspect_ratio_range=[1.0, 1.0],
        area_range=[0.5, 1.0],
        use_image_if_no_bounding_boxes=True,
        seed=seed)
    image = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    image.set_shape([None, None, target_image_shape[-1]])
  elif crop_method == "random":
    tf.set_random_seed(seed)
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = [h - size, w - size] * tf.random.uniform([2], 0, 1)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method == "middle":
    shape = tf.shape(image)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    image = tf.slice(image, begin, [size, size, 3])
  elif crop_method != "none":
    raise ValueError("Unsupported crop method: {}".format(crop_method))
  image = tf.image.resize_images(
      image, [target_image_shape[0], target_image_shape[1]])
  image.set_shape(target_image_shape)
  return image


@gin.configurable("train_imagenet_transform", whitelist=["crop_method"])
def _train_imagenet_transform(image, target_image_shape, seed,
                              crop_method="distorted"):
  return _transform_imagnet_image(
      image,
      target_image_shape=target_image_shape,
      crop_method=crop_method,
      seed=seed)


@gin.configurable("eval_imagenet_transform", whitelist=["crop_method"])
def _eval_imagenet_transform(image, target_image_shape, seed,
                             crop_method="middle"):
  return _transform_imagnet_image(
      image,
      target_image_shape=target_image_shape,
      crop_method=crop_method,
      seed=seed)


class ImagenetDataset(ImageDatasetV2):
  """ImageNet2012 as defined by TF Datasets."""

  def __init__(self, resolution, seed, filter_unlabeled=False):
    if resolution not in [64, 128, 256, 512]:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    super(ImagenetDataset, self).__init__(
        name="imagenet_{}".format(resolution),
        tfds_name="imagenet2012",
        resolution=resolution,
        colors=3,
        num_classes=1000,
        eval_test_samples=50000,
        seed=seed)
    self._eval_split = tfds.Split.VALIDATION
    self._filter_unlabeled = filter_unlabeled

  def _train_filter_fn(self, image, label):
    del image
    if not self._filter_unlabeled:
      return True
    logging.warning("Filtering unlabeled examples.")
    return tf.math.greater_equal(label, 0)

  def _train_transform_fn(self, image, label, seed):
    image = _train_imagenet_transform(
        image=image, target_image_shape=self.image_shape, seed=seed)
    return image, label

  def _eval_transform_fn(self, image, label, seed):
    image = _eval_imagenet_transform(
        image=image, target_image_shape=self.image_shape, seed=seed)
    return image, label


class SizeFilteredImagenetDataset(ImagenetDataset):
  """ImageNet from TFDS filtered by image size."""

  def __init__(self, resolution, threshold, seed):
    super(SizeFilteredImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "imagenet_{}_hq{}".format(resolution, threshold)
    self._threshold = threshold

  def _train_filter_fn(self, image, label):
    """The minimum image dimension has to be larger than the threshold."""
    del label
    size = tf.math.reduce_min(tf.shape(image)[:2])
    return tf.greater_equal(size, self._threshold)


class SingleClassImagenetDataset(ImagenetDataset):
  """ImageNet from TFDS with all instances having a constant label 0.

  It can be used to simmulate the setting where no labels are provided.
  """

  def __init__(self, resolution, seed):
    super(SingleClassImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "single_class_" + self._name
    self._num_classes = 1

  def _parse_fn(self, features):
    image, _ = super(SingleClassImagenetDataset, self)._parse_fn(features)
    label = tf.constant(0, dtype=tf.int32)
    return image, label


class RandomClassImagenetDataset(ImagenetDataset):
  """ImageNet2012 dataset with random labels."""

  def __init__(self, resolution, seed):
    super(RandomClassImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "random_class_" + self._name
    self._num_classes = 1000

  def _parse_fn(self, features):
    image, _ = super(RandomClassImagenetDataset, self)._parse_fn(features)
    label = tf.random.uniform(minval=0, maxval=1000, dtype=tf.int32)
    return image, label


class SoftLabeledImagenetDataset(ImagenetDataset):
  """ImageNet2012 dataset with soft labels."""

  def __init__(self, resolution, seed):
    super(SoftLabeledImagenetDataset, self).__init__(
        resolution=resolution,
        seed=seed)
    self._name = "soft_labeled_" + self._name

  def _replace_label(self, feature_dict, new_unparsed_label):
    """Replaces the label from the feature_dict with the new (soft) label.

    The function assumes that the new_unparsed_label contains a list of logits
    which will be converted to a soft label using the softmax.

    Args:
      feature_dict: A serialized TFRecord containing the old label.
      new_unparsed_label: A serialized TFRecord containing the new label.

    Returns:
      Updates the label in the label dict to the new soft label.
    """
    label_spec = {
        "file_name": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature([self._num_classes], tf.float32)
    }
    parsed_label = tf.parse_single_example(new_unparsed_label, label_spec)
    with tf.control_dependencies([
        tf.assert_equal(parsed_label["file_name"], feature_dict["file_name"])]):
      feature_dict["label"] = tf.nn.softmax(logits=parsed_label["label"])
    return feature_dict


DATASETS = {
    "celeb_a": CelebaDataset,
    "cifar10": Cifar10Dataset,
    "fashion-mnist": FashionMnistDataset,
    "lsun-bedroom": LsunBedroomDataset,
    "mnist": MnistDataset,
    "imagenet_64": functools.partial(ImagenetDataset, resolution=64),
    "imagenet_128": functools.partial(ImagenetDataset, resolution=128),
    "imagenet_256": functools.partial(ImagenetDataset, resolution=256),
    "imagenet_512": functools.partial(ImagenetDataset, resolution=512),
    "imagenet_512_hq400": (functools.partial(
        SizeFilteredImagenetDataset, resolution=512, threshold=400)),
    "danbooru_128": functools.partial(DanbooruDataset,resolution=128),
    "danbooru_256": functools.partial(DanbooruDataset,resolution=256),
    "danbooru_512": functools.partial(DanbooruDataset,resolution=512),
    "soft_labeled_imagenet_128": functools.partial(
        SoftLabeledImagenetDataset, resolution=128),
    "single_class_imagenet_128": functools.partial(
        SingleClassImagenetDataset, resolution=128),
    "random_class_imagenet_128": functools.partial(
        RandomClassImagenetDataset, resolution=128),
    "labeled_only_imagenet_128": functools.partial(
        ImagenetDataset, resolution=128, filter_unlabeled=True),
}


@gin.configurable("dataset")
def get_dataset(name, seed=547):
  """Instantiates a data set and sets the random seed."""
  if name not in DATASETS:
    raise ValueError("Dataset %s is not available." % name)
  return DATASETS[name](seed=seed)

