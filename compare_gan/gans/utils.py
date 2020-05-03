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

"""Utilities library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tensorflow as tf


def check_folder(log_dir):
  if not tf.gfile.IsDirectory(log_dir):
    tf.gfile.MakeDirs(log_dir)
  return log_dir


def save_images(images, image_path):
  with tf.gfile.Open(image_path, "wb") as f:
    scipy.misc.imsave(f, images * 255.0)


def rotate_images(images, rot90_scalars=(0, 1, 2, 3)):
  """Return the input image and its 90, 180, and 270 degree rotations."""
  images_rotated = [
      images,  # 0 degree
      tf.image.flip_up_down(tf.image.transpose_image(images)),  # 90 degrees
      tf.image.flip_left_right(tf.image.flip_up_down(images)),  # 180 degrees
      tf.image.transpose_image(tf.image.flip_up_down(images))  # 270 degrees
  ]

  results = tf.stack([images_rotated[i] for i in rot90_scalars])
  results = tf.reshape(results,
                       [-1] + images.get_shape().as_list()[1:])
  return results


def gaussian(batch_size, n_dim, mean=0., var=1.):
  return np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)

import gin

@gin.configurable(whitelist=["crop_method", "aspect_ratio_range", "area_range"])
def transform_image(image, target_image_shape, crop_method="distorted", seed=None,
                    aspect_ratio_range=[4.0 / 3.0, 3.0 / 4.0],
                    area_range=[0.08, 1.00],
                    ):
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
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
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
    image, [target_image_shape[0], target_image_shape[1]],
    method=tf.image.ResizeMethod.AREA)
  image.set_shape(target_image_shape)
  return image

def transform_images(images, **kws):
  shape = images.get_shape().as_list()
  if len(shape) < 4:
    crop = transform_image(images, shape, **kws)
  else:
    b, h, w, c = shape
    crop = tf.map_fn(lambda x: transform_image(x, [h, w, c], **kws), images)
  crop = tf.image.random_flip_left_right(crop)
  return crop

def lerp(a, b, t):
  return (b - a) * t + a

#from compare_gan.tpu import tpu_random

def random_crop_and_resize(images,
                           aspect_ratio_range = [4.0 / 3.0, 3.0 / 4.0],
                           area_range = [0.08, 1.00],
                           resize_method = tf.image.ResizeMethod.BILINEAR,
                           seed = None):
  b, h, w, c = images.get_shape().as_list()
  # def take_random_crop(img):
  #   u = tf.random.uniform((), minval=ratio, maxval=1.0)
  #   ch, cw = map(lambda x: tf.cast(x * ratio * u, dtype=tf.int32), (h, w))
  #   img = tf.random_crop(img, size=[ch, cw, 3])
  #   return img
  def take_random_crop(img):
    rmax, rmin = aspect_ratio_range
    amin, amax = area_range
    a = w * h
    area = tf.random.uniform((), amin * a, amax * a, seed=seed)
    #v = tpu_random.uniform(())
    #area = lerp(amin * a, amax * a, v)
    d = tf.math.sqrt(area)
    u = tf.random.uniform((), rmin, rmax, seed=seed)
    #u = lerp(rmin, rmax, tpu_random.uniform(()))
    fh = d * d / tf.math.sqrt(d * d / u);
    fw = tf.math.sqrt(d * d / u)
    ih = tf.cast(fh, tf.int32)
    iw = tf.cast(fw, tf.int32)
    #img = tf.random_crop(img, size=[ih, iw, 3])
    begin = [h - ih, w - iw] * tf.random.uniform([2], 0, 1)
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    img = tf.slice(img, begin, [ih, iw, 3])
    img = tf.image.resize_images(img, [h, w], method=resize_method)
    return img
  crop = tf.map_fn(take_random_crop, images)
  crop.set_shape([b, h, w, 3])
  #crop = tf.map_fn(lambda x: transform_image(x, [h, w, c]), images)
  crop = tf.image.random_flip_left_right(crop)
  return crop
