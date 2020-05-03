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


@gin.configurable
def random_brightness(images, max_delta=0.8):
  return tf.image.random_brightness(images, max_delta=max_delta)

@gin.configurable
def random_contrast(images, lower=0.2, upper=1.8):
  return tf.image.random_contrast(images, lower=lower, upper=upper)

@gin.configurable
def random_saturation(images, lower=0.2, upper=1.8):
  return tf.image.random_saturation(images, lower=lower, upper=upper)

@gin.configurable
def random_hue(images, max_delta=0.2):
  return tf.image.random_hue(images, max_delta=max_delta)

@gin.configurable
def clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0):
  return tf.clip_by_value(x, clip_value_min=clip_value_min, clip_value_max=clip_value_max)




def random_apply(func, p, x, seed=None):
  """Randomly apply function func to x with probability p."""
  if p == 1.0:
    return func(x)
  if p == 0.0:
    return x
  return tf.cond(
      tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32, seed=seed),
              tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)


def to_grayscale(image, keep_channels=True):
  image = tf.image.rgb_to_grayscale(image)
  if keep_channels:
    #image = tf.tile(image, [1, 1, 3])
    image = tf.image.grayscale_to_rgb(image)
  return image


@gin.configurable
def color_jitter(image,
                 strength,
                 brightness_strength=0.8,
                 contrast_strength=0.8,
                 saturation_strength=0.8,
                 hue_strength=0.2,
                 random_order=True,
                 seed=None):
  """Distorts the color of the image.
  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.
  Returns:
    The distorted image tensor.
  """
  brightness = 0 if brightness_strength == 0 else brightness_strength * strength
  contrast = 0 if contrast_strength == 0 else contrast_strength * strength
  saturation = 0 if saturation_strength == 0 else saturation_strength * strength
  hue = 0 if hue_strength == 0 else hue_strength * strength
  if random_order:
    return color_jitter_rand(image, brightness, contrast, saturation, hue, seed=seed)
  else:
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue)




def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is fixed).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = tf.image.random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf.clip_by_value(image, 0., 1.)
    return image



def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0, seed=None):
  """Distorts the color of the image (jittering order is random).
  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.
  Returns:
    The distorted image tensor.
  """
  with tf.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return tf.image.random_brightness(x, max_delta=brightness, seed=seed)
      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf.image.random_contrast(x, lower=1-contrast, upper=1+contrast, seed=seed)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation, seed=seed)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf.image.random_hue(x, max_delta=hue, seed=seed)
      x = tf.cond(tf.less(i, 2),
                  lambda: tf.cond(tf.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf.cond(tf.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf.random_shuffle(tf.range(4), seed=seed)
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf.clip_by_value(image, 0., 1.)
    return image


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.
  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf.to_int32(kernel_size / 2)
  kernel_size = radius * 2 + 1
  x = tf.to_float(tf.range(-radius, radius + 1))
  blur_filter = tf.exp(
      -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.to_float(sigma), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


import functools

@gin.configurable
def random_color_jitter(image, p=1.0,
                        color_jitter_strength=1.0,
                        color_jitter_chance=0.8,
                        color_drop_chance=0.2,
                        seed=None):
  def _transform(image):
    color_jitter_t = functools.partial(
        color_jitter, strength=color_jitter_strength, seed=seed)
    image = random_apply(color_jitter_t, p=color_jitter_chance, x=image, seed=seed)
    return random_apply(to_grayscale, p=color_drop_chance, x=image, seed=seed)
  return random_apply(_transform, p=p, x=image, seed=seed)


@gin.configurable(blacklist=["image", "seed"])
def random_blur(image, image_size=gin.REQUIRED, p=1.0, seed=None):
  """Randomly blur an image.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.
  Returns:
    A preprocessed image `Tensor`.
  """
  def _transform(image):
    sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32, seed=seed)
    return gaussian_blur(
        image, kernel_size=image_size//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image, seed=seed)


@gin.configurable(blacklist=["images", "seed"])
def transform(images, crop=True, flip=True, color=True, clip=True, blur=False, seed=None):
  if crop:
    images = transform_images(images, seed=seed)
  if blur:
    images = random_blur(images, seed=seed)
  if flip:
    images = tf.image.random_flip_left_right(images)
  if color:
    images = random_color_jitter(images, seed=seed)
  if clip:
    images = clip_by_value(images)
  return images
