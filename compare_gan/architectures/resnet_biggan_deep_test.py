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

"""Test number of parameters for the BigGAN-Deep architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from compare_gan import utils
from compare_gan.architectures import arch_ops
from compare_gan.architectures import resnet_biggan_deep
import tensorflow as tf


class ResNet5BigGanDeepTest(tf.test.TestCase):

  def testNumberOfParameters(self):
    with tf.Graph().as_default():
      batch_size = 1
      z = tf.zeros((batch_size, 128))
      y = tf.one_hot(tf.ones((batch_size,), dtype=tf.int32), 1000)
      options = {'plain_tanh': True}
      generator = resnet_biggan_deep.Generator(
          image_shape=(512, 512, 3),
          batch_norm_fn=arch_ops.conditional_batch_norm,
          ch=128,
          spectral_norm=True,
          experimental_fast_conv_to_rgb=True,
          options=options,
          )
      fake_images = generator(z, y=y, is_training=True, reuse=False)
      self.assertEqual(fake_images.shape.as_list(), [batch_size, 512, 512, 3])
      discriminator = resnet_biggan_deep.Discriminator(
          ch=128,
          spectral_norm=True,
          )
      predictions = discriminator(fake_images, y, is_training=True)
      self.assertLen(predictions, 3)

      t_vars = tf.trainable_variables()
      l_vars = [x for x in tf.global_variables() + tf.local_variables() if not x.trainable]
      for i, a_vars in enumerate([l_vars, t_vars]):
        g_vars = [var for var in a_vars if "generator" in var.name]
        d_vars = [var for var in a_vars if "discriminator" in var.name]
        g_param_overview = utils.get_parameter_overview(g_vars, limit=None)
        d_param_overview = utils.get_parameter_overview(d_vars, limit=None)
        g_param_overview = g_param_overview.split("\n")
        tag = "(local)" if i == 1 else ""
        logging.info("Generator variables %s:", tag)
        for i in range(0, len(g_param_overview), 80):
          logging.info("\n%s", "\n".join(g_param_overview[i:i + 80]))
        logging.info("Discriminator variables %s:\n%s", tag, d_param_overview)

      g_num_weights = sum([v.get_shape().num_elements() for v in g_vars])
      #self.assertEqual(g_num_weights, 50244484)
      self.assertEqual(g_num_weights, 56163905)

      d_num_weights = sum([v.get_shape().num_elements() for v in d_vars])
      #self.assertEqual(d_num_weights, 34590210)
      self.assertEqual(d_num_weights, 38301122)


if __name__ == "__main__":
  tf.test.main()
