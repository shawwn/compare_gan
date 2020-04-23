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

"""Implementation of Self-Supervised GAN with contrastive loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
from compare_gan.architectures.arch_ops import linear
from compare_gan.gans import loss_lib
from compare_gan.gans import modular_gan
from compare_gan.gans import penalty_lib
from compare_gan.gans import utils

import gin
import numpy as np
import random
import tensorflow as tf

FLAGS = flags.FLAGS

# augmentation functions

# augment

def random_crop_and_resize(images, ratio=0.8):
  b, h, w, c = images.get_shape().as_list()
  ch, cw = map(lambda x: int(x * ratio), (h, w))
  crop = tf.random_crop(images, size=[b, ch, cw, 3])
  crop = tf.image.resize(crop, [h, w])
  return crop

def random_apply(fn, image, prob=1.):
  if random.random() > prob:
    return image
  return fn(image)

def color_distortion(image, s=1.0):
  lower, upper, x = (1 - 0.8 * s), (1 + 0.8 * s), image
  x = tf.image.random_brightness(x, max_delta=0.8*s)
  x = tf.image.random_contrast(x, lower=lower, upper=upper)
  x = tf.image.random_saturation(x, lower=lower, upper=upper)
  x = tf.image.random_hue(x, max_delta=0.2*s)
  x = tf.clip_by_value(x, 0, 1)
  return x

def color_drop(image):
  image = tf.image.rgb_to_grayscale(image)
  image = tf.tile(image, [1, 1, 3])
  return image

# pylint: disable=not-callable
@gin.configurable(blacklist=["kwargs"])
class CLGAN(modular_gan.ModularGAN):
  """Self-Supervised GAN with Contrastive Loss"""

  def __init__(self,
               aug_color_jitter_prob=0.8,
               aug_color_drop_prob=0.2,
               weight_contrastive_loss_d=10.0,
               **kwargs):
    """Creates a new Self-Supervised GAN using Contrastive Loss.

    Args:
      self_supervised_batch_size: The total number images per batch for the self supervised loss.
      weight_contrastive_loss_d: Weight for the contrastive loss for the self supervised learning on real images
      **kwargs: Additional arguments passed to `ModularGAN` constructor.
    """
    super(CLGAN, self).__init__(**kwargs)

    self._weight_contrastive_loss_d = weight_contrastive_loss_d
    self._aug_color_jitter_prob = aug_color_jitter_prob
    self._aug_color_drop_prob = aug_color_drop_prob

    # To safe memory ModularGAN supports feeding real and fake samples
    # separately through the discriminator. CLGAN does not support this to
    # avoid additional additional complexity in create_loss().
    assert not self._deprecated_split_disc_calls, \
        "Splitting discriminator calls is not supported in CLGAN."

  def _latent_projections(self, latents):
    bs, dim = latents.get_shape().as_list()

    with tf.variable_scope("discriminator_z_projection", reuse=tf.AUTO_REUSE) as scope:
      k1 = tf.get_variable("kernel1", [dim, dim * 4])
      k2 = tf.get_variable("kernel2", [dim * 4, dim])
      z_proj = tf.matmul(tf.nn.leaky_relu(tf.matmul(latents, k1), name=scope.name), k2)
      z_proj = z_proj / tf.reshape(tf.norm(z_proj, ord=2, axis=-1), [bs, 1])
      return z_proj

  def create_loss(self, features, labels, params, is_training=True):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. These are class indices. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    images = features["images"]  # Input images.
    generated = features["generated"]  # Fake images.
    if self.conditional:
      y = self._get_one_hot_labels(labels)
      sampled_y = self._get_one_hot_labels(features["sampled_labels"])
    else:
      y = None
      sampled_y = None
      all_y = None

    # Batch size per core.
    bs = images.shape[0].value

    aug_images = random_crop_and_resize(images)
    aug_images = random_apply(color_distortion, aug_images, self._aug_color_jitter_prob)
    aug_images = random_apply(color_drop, aug_images, self._aug_color_drop_prob)

    # concat all images
    all_images = tf.concat([images, generated, aug_images], 0)

    if self.conditional:
      all_y = tf.concat([y, sampled_y, y], axis=0)

    # Compute discriminator output for real and fake images in one batch.

    d_all, d_all_logits, d_latents = self.discriminator(
        x=all_images, y=all_y, is_training=is_training)

    z_projs = self._latent_projections(d_latents)

    d_real, d_fake, _ = tf.split(d_all, 3)
    d_real_logits, d_fake_logits, _ = tf.split(d_all_logits, 3)
    z_projs_real, _, z_aug_projs_real = tf.split(z_projs, 3)

    self.d_loss, _, _, self.g_loss = loss_lib.get_losses(
        d_real=d_real, d_fake=d_fake, d_real_logits=d_real_logits,
        d_fake_logits=d_fake_logits)

    penalty_loss = penalty_lib.get_penalty_loss(
        x=images, x_fake=generated, y=y, is_training=is_training,
        discriminator=self.discriminator, architecture=self._architecture)
    self.d_loss += self._lambda * penalty_loss

    sims_logits = tf.matmul(z_projs_real, z_aug_projs_real, transpose_b=True)    
    sims_probs = tf.nn.softmax(sims_logits)

    sim_labels = tf.constant(np.arange(bs, dtype=np.int32))
    sims_onehot = tf.one_hot(sim_labels, bs)

    c_real_loss = - tf.reduce_mean(
        tf.reduce_sum(sims_onehot * tf.log(sims_probs + 1e-10), 1))

    self.d_loss += c_real_loss * self._weight_contrastive_loss_d

    self._tpu_summary.scalar("loss/c_real_loss", c_real_loss)
    self._tpu_summary.scalar("loss/penalty", penalty_loss)

