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

"""Re-implementation of BigGAN architecture.

Disclaimer: We note that this is our best-effort re-implementation and stress
that even minor implementation differences may lead to large differences in
trained models due to sensitivity of GANs to optimization hyperparameters and
details of neural architectures. That being said, this code suffices to
reproduce the reported FID on ImageNet 128x128.

Based on "Large Scale GAN Training for High Fidelity Natural Image Synthesys",
Brock A. et al., 2018 [https://arxiv.org/abs/1809.11096].

Supported resolutions: 32, 64, 128, 256, 512, 1024. The location of the self-attention
block must be set in the Gin config. See below.

Notable differences to resnet5.py:
- Much wider layers by default.
- 1x1 convs for shortcuts in D and G blocks.
- Last BN in G is unconditional.
- tanh activation in G.
- No shortcut in D block if in_channels == out_channels.
- sum pooling instead of mean pooling in D.
- Last block in D does not downsample.

Information related to parameter counts and Gin configuration:
128x128
-------
Number of parameters: (D) 87,982,370 (G) 70,433,988
Required Gin settings:
options.z_dim = 120
resnet_biggan.Generator.blocks_with_attention = "B4"
resnet_biggan.Discriminator.blocks_with_attention = "B1"

256x256
-------
Number of parameters: (D) 98,635,298 (G) 82,097,604
Required Gin settings:
options.z_dim = 140
resnet_biggan.Generator.blocks_with_attention = "B5"
resnet_biggan.Discriminator.blocks_with_attention = "B2"

512x512
-------
Number of parameters: (D)  98,801,378 (G) 82,468,068
Required Gin settings:
options.z_dim = 160
resnet_biggan.Generator.blocks_with_attention = "B4"
resnet_biggan.Discriminator.blocks_with_attention = "B3"

1024x1024
-------
Number of parameters: (D)  98,801,378 (G) 82,468,068
Required Gin settings:
options.z_dim = 180
resnet_biggan.Generator.blocks_with_attention = "B4"
resnet_biggan.Discriminator.blocks_with_attention = "B4"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops as ops
from compare_gan.architectures import resnet_ops
from compare_gan.architectures import stylegan_ops

import gin
from six.moves import range
import tensorflow as tf


@gin.configurable(blacklist=['use_relu'])
class BigGanResNetBlock(resnet_ops.ResNetBlock):
  """ResNet block with options for various normalizations.

  This block uses a 1x1 convolution for the (optional) shortcut connection.
  """

  def __init__(self,
               add_shortcut=True,
               use_relu=True,
               **kwargs):
    """Constructs a new ResNet block for BigGAN.

    Args:
      add_shortcut: Whether to add a shortcut connection.
      **kwargs: Additional arguments for ResNetBlock.
    """
    super(BigGanResNetBlock, self).__init__(**kwargs)
    self._add_shortcut = add_shortcut
    self._use_relu = use_relu

  def apply(self, inputs, z, y, is_training):
    """"ResNet block containing possible down/up sampling, shared for G / D.

    Args:
      inputs: a 3d input tensor of feature map.
      z: the latent vector for potential self-modulation. Can be None if use_sbn
        is set to False.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, whether or notthis is called during the training.

    Returns:
      output: a 3d output tensor of feature map.
    """
    if inputs.shape[-1].value != self._in_channels:
      raise ValueError(
          "Unexpected number of input channels (expected {}, got {}).".format(
              self._in_channels, inputs.shape[-1].value))

    with tf.variable_scope(self._name, values=[inputs]):
      outputs = inputs

      outputs = self.batch_norm(
          outputs, z=z, y=y, is_training=is_training, name="bn1")
      if self._layer_norm:
        logging.info("[Block] %s using layer_norm", inputs.shape)
        outputs = ops.layer_norm(outputs, is_training=is_training, scope="ln1")

      if self._use_relu:
        outputs = tf.nn.relu(outputs)
      else:
        logging.info("[Block] %s skipping relu", inputs.shape)
      outputs = self._get_conv(
          outputs, self._in_channels, self._out_channels, self._scale1,
          suffix="conv1")

      outputs = self.batch_norm(
          outputs, z=z, y=y, is_training=is_training, name="bn2")
      if self._layer_norm:
        outputs = ops.layer_norm(outputs, is_training=is_training, scope="ln2")

      if self._use_relu:
        outputs = tf.nn.relu(outputs)
      outputs = self._get_conv(
          outputs, self._out_channels, self._out_channels, self._scale2,
          suffix="conv2")

      # Combine skip-connection with the convolved part.
      if self._add_shortcut:
        shortcut = self._get_conv(
            inputs, self._in_channels, self._out_channels, self._scale,
            kernel_size=(1, 1),
            suffix="conv_shortcut")
        outputs += shortcut
      logging.info("[Block] %s (z=%s, y=%s) -> %s", inputs.shape,
                   None if z is None else z.shape,
                   None if y is None else y.shape, outputs.shape)
      return outputs


@gin.configurable
class Generator(abstract_arch.AbstractGenerator):
  """ResNet-based generator supporting resolutions 32, 64, 128, 256, 512, 1024."""

  def __init__(self,
               ch=96,
               blocks_with_attention="64",
               stylegan_z=False,
               hierarchical_z=True,
               embed_z=False,
               embed_y=True,
               embed_y_dim=128,
               embed_bias=False,
               channel_multipliers=None,
               plain_tanh=False,
               use_noise=False,
               randomize_noise=True,
               **kwargs):
    """Constructor for BigGAN generator.

    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      hierarchical_z: Split z into chunks and only give one chunk to each.
        Each chunk will also be concatenated to y, the one hot encoded labels.
      embed_z: If True use a learnable embedding of z that is used instead.
        The embedding will have the length of z.
      embed_y: If True use a learnable embedding of y that is used instead.
      embed_y_dim: Size of the embedding of y.
      embed_bias: Use bias with for the embedding of z and y.
      **kwargs: additional arguments past on to ResNetGenerator.
    """
    super(Generator, self).__init__(**kwargs)
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._blocks_with_attention.discard('')
    self._channel_multipliers = None if channel_multipliers is None else [int(x.strip()) for x in channel_multipliers.split(",")]
    self._hierarchical_z = hierarchical_z
    self._stylegan_z = stylegan_z
    self._embed_z = embed_z
    self._embed_y = embed_y
    self._embed_y_dim = embed_y_dim
    self._embed_bias = embed_bias
    self._plain_tanh = self.options.get('plain_tanh', plain_tanh)
    bn_activation = self.options.get('bn_activation', 'relu')
    assert bn_activation in ['none', 'relu']
    self._use_relu = bn_activation == 'relu'
    self._use_noise = use_noise
    self._randomize_noise = randomize_noise
    if hierarchical_z and stylegan_z:
      raise ValueError("Must set either hierarchical_z or stylegan_z, not both")

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm,
        use_relu=self._use_relu)

  def _get_in_out_channels(self):
    resolution = self._image_shape[0]
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    elif resolution == 1024:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1, 1]
    elif resolution == 512:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1]
    elif resolution == 256:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1]
    elif resolution == 128:
      channel_multipliers = [16, 16, 8, 4, 2, 1]
    elif resolution == 64:
      channel_multipliers = [16, 16, 8, 4, 2]
    elif resolution == 32:
      channel_multipliers = [4, 4, 4, 4]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self._ch * c for c in channel_multipliers[:-1]]
    out_channels = [self._ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels

  @property
  @gin.configurable("Generator_z_args")
  def G_main_args(self, **args):
    return args

  def apply(self, z, y, *, is_training, **kwds):
    """Build the generator network for the given inputs.

    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, are we in train or eval model.

    Returns:
      A tensor of size [batch_size] + self._image_shape with values in [0, 1].
    """
    with gin.config_scope("generator"):
      return self._apply(z, y, is_training=is_training, **kwds)

  def _apply(self, z, y, *, is_training, truncation=None, **kwds):
    shape_or_none = lambda t: None if t is None else t.shape
    logging.info("[Generator] inputs are z=%s, y=%s", z.shape, shape_or_none(y))
    # Each block upscales by a factor of 2.
    seed_size = 4
    z_dim = z.shape[1].value

    in_channels, out_channels = self._get_in_out_channels()
    num_blocks = len(in_channels)

    if self._embed_z:
      z = ops.linear(z, z_dim, scope="embed_z", use_sn=False,
                     use_bias=self._embed_bias)
    if self._embed_y:
      y = ops.linear(y, self._embed_y_dim, scope="embed_y", use_sn=False,
                     use_bias=self._embed_bias)
    if truncation is not None:
      z *= truncation
      y *= truncation
    if self._stylegan_z:
      z_args = self.G_main_args
      z_args['is_training'] = z_args.pop('is_training', is_training)
      tf.logging.info('[Generator] scope: %s z_args: %s', gin.current_scope_str(), z_args)
      z_per_block = stylegan_ops.G_main(num_blocks + 1, z, None, latent_size=z_dim, **z_args)
      z_per_block = tf.unstack(z_per_block, axis=1)
      z0, z_per_block = z_per_block[0], z_per_block[1:]
      if y is not None:
        y_per_block = [tf.concat([zi, y], 1) for zi in z_per_block]
    elif self._hierarchical_z:
      z_per_block = tf.split(z, num_blocks + 1, axis=1)
      z0, z_per_block = z_per_block[0], z_per_block[1:]
      if y is not None:
        y_per_block = [tf.concat([zi, y], 1) for zi in z_per_block]
    else:
      z0 = z
      z_per_block = num_blocks * [z]
      y_per_block = num_blocks * [y]

    logging.info("[Generator] z0=%s, z_per_block=%s, y_per_block=%s",
                 z0.shape, [str(shape_or_none(t)) for t in z_per_block],
                 [str(shape_or_none(t)) for t in y_per_block])

    # Map noise to the actual seed.
    net = ops.linear(
        z0,
        in_channels[0] * seed_size * seed_size,
        scope="fc_noise",
        use_sn=self._spectral_norm)
    # Reshape the seed to be a rank-4 Tensor.
    net = tf.reshape(
        net,
        [-1, seed_size, seed_size, in_channels[0]],
        name="fc_reshaped")

    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      if self._use_noise:
        net = ops.noise_block(net, name=name, randomize_noise=self._randomize_noise)
      block = self._resnet_block(
          name=name,
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale="up")
      net = block(
          net,
          z=z_per_block[block_idx],
          y=y_per_block[block_idx],
          is_training=is_training)
      res = net.shape[1].value
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        logging.info("[Generator] Applying non-local block at %dx%d resolution to %s",
                     res, res, net.shape)
        net = ops.non_local_block(net, "non_local_block",
                                  use_sn=self._spectral_norm)
    assert len(blocks_with_attention) <= 0

    # Final processing of the net.
    # Use unconditional batch norm.
    logging.info("[Generator] before final processing: %s", net.shape)
    if self._use_noise:
      net = ops.noise_block(net, name="final_norm", randomize_noise=self._randomize_noise)
    net = ops.batch_norm(net, is_training=is_training, name="final_norm")
    if self._use_relu:
      net = tf.nn.relu(net)
    else:
      logging.info("[Generator] skipping relu")
    net = ops.conv2d(net, output_dim=self._image_shape[2], k_h=3, k_w=3,
                     d_h=1, d_w=1, name="final_conv",
                     use_sn=self._spectral_norm)
    logging.info("[Generator] after final processing: %s (plain_tanh=%s)", net.shape, self._plain_tanh)
    if self._plain_tanh:
      net = tf.nn.tanh(net)
    else:
      net = (tf.nn.tanh(net) + 1.0) / 2.0
    return net


@gin.configurable
class Discriminator(abstract_arch.AbstractDiscriminator):
  """ResNet-based discriminator supporting resolutions 32, 64, 128, 256, 512, 1024."""

  def __init__(self,
               ch=96,
               blocks_with_attention="64",
               project_y=True,
               channel_multipliers=None,
               use_noise=False,
               randomize_noise=True,
               **kwargs):
    """Constructor for BigGAN discriminator.

    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      project_y: Add an embedding of y in the output layer.
      **kwargs: additional arguments past on to ResNetDiscriminator.
    """
    super(Discriminator, self).__init__(**kwargs)
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._blocks_with_attention.discard('')
    self._channel_multipliers = None if channel_multipliers is None else [int(x.strip()) for x in channel_multipliers.split(",")]
    self._project_y = project_y
    self._use_noise = use_noise
    self._randomize_noise = randomize_noise

  def _resnet_block(self, name, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        name=name,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        add_shortcut=in_channels != out_channels,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)

  def _get_in_out_channels(self, colors, resolution):
    if colors not in [1, 3]:
      raise ValueError("Unsupported color channels: {}".format(colors))
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    elif resolution == 1024:
      channel_multipliers = [1, 1, 1, 2, 4, 8, 8, 16, 16]
    elif resolution == 512:
      channel_multipliers = [1, 1, 2, 4, 8, 8, 16, 16]
    elif resolution == 256:
      channel_multipliers = [1, 2, 4, 8, 8, 16, 16]
    elif resolution == 128:
      channel_multipliers = [1, 2, 4, 8, 16, 16]
    elif resolution == 64:
      channel_multipliers = [2, 4, 8, 16, 16]
    elif resolution == 32:
      channel_multipliers = [2, 2, 2, 2]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    out_channels = [self._ch * c for c in channel_multipliers]
    in_channels = [colors] + out_channels[:-1]
    return in_channels, out_channels

  def apply(self, x, y, is_training):
    """Apply the discriminator on a input.

    Args:
      x: `Tensor` of shape [batch_size, ?, ?, ?] with real or fake images.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: Boolean, whether the architecture should be constructed for
        training or inference.

    Returns:
      Tuple of 3 Tensors, the final prediction of the discriminator, the logits
      before the final output activation function and logits form the second
      last layer.
    """
    with gin.config_scope("discriminator"):
      return self._apply(x, y, is_training)

  def _apply(self, x, y, is_training):
    logging.info("[Discriminator] inputs are x=%s, y=%s", x.shape,
                 None if y is None else y.shape)
    resnet_ops.validate_image_inputs(x)

    in_channels, out_channels = self._get_in_out_channels(
        colors=x.shape[-1].value, resolution=x.shape[1].value)
    num_blocks = len(in_channels)

    net = x
    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      if self._use_noise:
        net = ops.noise_block(net, name=name, randomize_noise=self._randomize_noise)
      is_last_block = block_idx == num_blocks - 1
      block = self._resnet_block(
          name=name,
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale="none" if is_last_block else "down")
      net = block(net, z=None, y=y, is_training=is_training)
      res = net.shape[1].value
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        logging.info("[Discriminator] Applying non-local block at %dx%d resolution to %s",
                     res, res, net.shape)
        net = ops.non_local_block(net, "non_local_block",
                                  use_sn=self._spectral_norm)
    assert len(blocks_with_attention) <= 0

    # Final part
    logging.info("[Discriminator] before final processing: %s", net.shape)
    if self._use_noise:
      net = ops.noise_block(net, name="final_fc", randomize_noise=self._randomize_noise)
    net = tf.nn.relu(net)
    h = tf.math.reduce_sum(net, axis=[1, 2])
    out_logit = ops.linear(h, 1, scope="final_fc", use_sn=self._spectral_norm)
    logging.info("[Discriminator] after final processing: %s", net.shape)
    if self._project_y:
      if y is None:
        raise ValueError("You must provide class information y to project.")
      with tf.variable_scope("embedding_fc"):
        y_embedding_dim = out_channels[-1]
        # We do not use ops.linear() below since it does not have an option to
        # override the initializer.
        kernel = tf.get_variable(
            "kernel", [y.shape[1], y_embedding_dim], tf.float32,
            initializer=tf.initializers.glorot_normal())
        kernel = ops.graph_spectral_norm(kernel)
        if self._spectral_norm:
          kernel, norm = ops.spectral_norm(kernel)
        embedded_y = tf.matmul(y, kernel)
        logging.info("[Discriminator] embedded_y for projection: %s",
                     embedded_y.shape)
        out_logit += tf.reduce_sum(embedded_y * h, axis=1, keepdims=True)
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit, h
