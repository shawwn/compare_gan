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

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import

def tpu_cross_replica_concat(tensor, tpu_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    tpu_context: A `TPUContext`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if tpu_context is None or tpu_context.num_replicas <= 1:
    return tensor

  num_replicas = tpu_context.num_replicas

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])


def upfirdn_2d(x, k, upx=1, upy=1, downx=1, downy=1, padx0=0, padx1=0, pady0=0, pady1=0, impl='ref', padding='VALID',
               data_format='NCHW'):
  r"""Pad, upsample, FIR filter, and downsample a batch of 2D images.

  Accepts a batch of 2D images of the shape `[majorDim, inH, inW, minorDim]`
  and performs the following operations for each image, batched across
  `majorDim` and `minorDim`:

  1. Pad the image with zeros by the specified number of pixels on each side
     (`padx0`, `padx1`, `pady0`, `pady1`). Specifying a negative value
     corresponds to cropping the image.

  2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`).

  3. Convolve the image with the specified 2D FIR filter (`k`), shrinking the
     image so that the footprint of all output pixels lies within the input image.

  4. Downsample the image by throwing away pixels (`downx`, `downy`).

  This sequence of operations bears close resemblance to scipy.signal.upfirdn().
  The fused op is considerably more efficient than performing the same calculation
  using standard TensorFlow ops. It supports gradients of arbitrary order.

  Args:
      x:      Input tensor of the shape `[majorDim, inH, inW, minorDim]`.
      k:      2D FIR filter of the shape `[firH, firW]`.
      upx:    Integer upsampling factor along the X-axis (default: 1).
      upy:    Integer upsampling factor along the Y-axis (default: 1).
      downx:  Integer downsampling factor along the X-axis (default: 1).
      downy:  Integer downsampling factor along the Y-axis (default: 1).
      padx0:  Number of pixels to pad on the left side (default: 0).
      padx1:  Number of pixels to pad on the right side (default: 0).
      pady0:  Number of pixels to pad on the top side (default: 0).
      pady1:  Number of pixels to pad on the bottom side (default: 0).
      impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

  Returns:
      Tensor of the shape `[majorDim, outH, outW, minorDim]`, and same datatype as `x`.
  """

  impl_dict = {
    'ref': _upfirdn_2d_ref,
  }
  return impl_dict[impl](x=x, k=k, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0,
                         pady1=pady1, padding=padding, data_format=data_format)


def _upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1, padding, data_format, cpu=False):
  """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops."""

  x = tf.convert_to_tensor(x)
  k = np.asarray(k, dtype=np.float32)
  assert x.shape.rank == 4
  inH = x.shape[1].value
  inW = x.shape[2].value
  minorDim = _shape(x, 3)
  kernelH, kernelW = k.shape
  assert inW >= 1 and inH >= 1
  assert kernelW >= 1 and kernelH >= 1
  assert isinstance(upx, int) and isinstance(upy, int)
  assert isinstance(downx, int) and isinstance(downy, int)
  assert isinstance(padx0, int) and isinstance(padx1, int)
  assert isinstance(pady0, int) and isinstance(pady1, int)

  # Upsample (insert zeros).
  x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
  x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
  x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

  # Pad (crop if negative).
  x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
  x = x[:, max(-pady0, 0): x.shape[1].value - max(-pady1, 0), max(-padx0, 0): x.shape[2].value - max(-padx1, 0), :]

  # Convolve with filter.
  x = tf.transpose(x, [0, 3, 1, 2])
  x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])
  w = tf.constant(k[::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)
  if cpu:
    x = _o(tf.nn.conv2d(_i(x), w, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC'))
  else:
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW')
  x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
  x = tf.transpose(x, [0, 2, 3, 1])

  # Downsample (throw away pixels).
  return x[:, ::downy, ::downx, :]


def _shape(tf_expr, dim_idx):
  if tf_expr.shape.rank is not None:
    dim = tf_expr.shape[dim_idx].value
    if dim is not None:
      return dim
  return tf.shape(tf_expr)[dim_idx]


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  if k.ndim == 1:
    k = np.outer(k, k)
  k /= np.sum(k)
  assert k.ndim == 2
  assert k.shape[0] == k.shape[1]
  return k


def _simple_upfirdn_2d(x, k, up=1, down=1, pad0=0, pad1=0, padding='VALID', data_format='NCHW', impl='ref'):
  assert data_format in ['NCHW', 'NHWC']
  assert x.shape.rank == 4
  y = x
  if data_format == 'NCHW':
    y = tf.reshape(y, [-1, _shape(y, 2), _shape(y, 3), 1])
  y = upfirdn_2d(y, k, upx=up, upy=up, downx=down, downy=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1,
                 padding=padding, data_format=data_format, impl=impl)
  if data_format == 'NCHW':
    y = tf.reshape(y, [-1, _shape(x, 1), _shape(y, 1), _shape(y, 2)])
  return y


def downsample_2d(x, k=None, factor=2, gain=1, data_format='NCHW', impl='ref'):
  r"""Downsample a batch of 2D images with the given filter.

  Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]`
  and downsamples each image with the given filter. The filter is normalized so that
  if the input pixels are constant, they will be scaled by the specified `gain`.
  Pixels outside the image are assumed to be zero, and the filter is padded with
  zeros so that its shape is a multiple of the downsampling factor.

  Args:
      x:            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
      k:            FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                    The default is `[1] * factor`, which corresponds to average pooling.
      factor:       Integer downsampling factor (default: 2).
      gain:         Scaling factor for signal magnitude (default: 1.0).
      data_format:  `'NCHW'` or `'NHWC'` (default: `'NCHW'`).
      impl:         Name of the implementation to use. Can be `"ref"` (default) or `"cuda"`.

  Returns:
      Tensor of the shape `[N, C, H // factor, W // factor]` or
      `[N, H // factor, W // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor
  return _simple_upfirdn_2d(x, k, down=factor, pad0=(p + 1) // 2, pad1=p // 2, data_format=data_format, impl=impl)


def _centered(arr, newshape):
  # Return the center newshape portion of the array.
  currshape = tf.shape(arr)[-2:]
  startind = (currshape - newshape) // 2
  endind = startind + newshape
  return arr[..., startind[0]:endind[0], startind[1]:endind[1]]


def tf_fftconv(in1, in2, mode="full"):
  # Reorder channels to come second (needed for fft)
  #in1 = tf.transpose(in1, perm=[0, 3, 1, 2])
  #in2 = tf.transpose(in2, perm=[0, 3, 1, 2])
  # Extract shapes
  s1 = tf.convert_to_tensor(tf.shape(in1)[-2:])
  s2 = tf.convert_to_tensor(tf.shape(in2)[-2:])
  #shape = s1 + s2 - 1
  shape = s2
  # Compute convolution in fourier space
  sp1 = tf.signal.rfft2d(in1, shape)
  sp2 = tf.signal.rfft2d(in2, shape)
  ret = tf.signal.irfft2d(sp1 * sp2, shape) # not implemented on TPUs
  #ret = tf.signal.irfft(tf.linalg.matrix_transpose(tf.signal.ifft(tf.linalg.matrix_transpose(sp1 * sp2))))
  # Crop according to mode
  if mode == "full":
    cropped = ret
  elif mode == "same":
    cropped = _centered(ret, s1)
  elif mode == "valid":
    cropped = _centered(ret, s1 - s2 + 1)
  else:
    raise ValueError("Acceptable mode flags are 'valid',"
                     " 'same', or 'full'.")
  # Reorder channels to last
  #result = tf.transpose(cropped, perm=[0, 2, 3, 1])
  result = cropped
  return result

def tf_fftconv(a, b, mode, cpu=False):
  if cpu:
    a = tf.transpose(a, [0, 2, 3, 1])
  r = tf.nn.conv2d(a, b, [1, 1, 1, 1], 'VALID', data_format=('NHWC' if cpu else 'NCHW'))
  if cpu:
    r = tf.transpose(r, [0, 3, 1, 2])
  return r

from tensorflow.python.ops import array_ops, math_ops

def tf_fspecial_gauss(size, sigma):
    y, x = array_ops.meshgrid(math_ops.range(size)-5, math_ops.range(size)-5);
    x = tf.cast(x[..., None, None], dtype=tf.float32)
    y = tf.cast(y[..., None, None], dtype=tf.float32)
    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    r = g / tf.reduce_sum(g)
    return r

def tf_gaussian(std=20, sigma=1.5):
    grid_x, grid_y = array_ops.meshgrid(math_ops.range(3 * std), math_ops.range(3 * std))
    grid_x = tf.cast(grid_x[..., None, None], 'float32')
    grid_y = tf.cast(grid_y[..., None, None], 'float32')
    gaussian = tf.exp(-((grid_x - sigma * std) ** 2 + (grid_y - sigma * std) ** 2) / std ** 2)
    gaussian = gaussian / tf.reduce_sum(gaussian)
    return gaussian


# https://stackoverflow.com/questions/47272699/need-tensorflow-keras-equivalent-for-scipy-signal-fftconvolve

def tf_ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, mode='full', cs_map=False):
  window = tf_fspecial_gauss(filter_size, filter_sigma)
  C1 = (k1*max_val)**2
  C2 = (k2*max_val)**2
  mu1 = tf_fftconv(img1, window, mode=mode)
  mu2 = tf_fftconv(img2, window, mode=mode)
  mu1_sq = mu1*mu1
  mu2_sq = mu2*mu2
  mu1_mu2 = mu1*mu2
  sigma1_sq = tf_fftconv(img1*img1, window, mode=mode) - mu1_sq
  sigma2_sq = tf_fftconv(img2*img2, window, mode=mode) - mu2_sq
  sigma12 = tf_fftconv(img1*img2, window, mode=mode) - mu1_mu2
  if cs_map:
    return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                (sigma1_sq + sigma2_sq + C2)),
            (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
  else:
    return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                (sigma1_sq + sigma2_sq + C2))

import gin

# https://github.com/tensorflow/models/blob/394baa9f21424d3522ccfbdcee8acd3840891ff6/research/compression/image_encoder/msssim.py

@gin.configurable(whitelist=['filter_size', 'filter_sigma', 'k1', 'k2', 'weights'])
def tf_ssim_multiscale(img1, img2, max_val=1.0,
    filter_size=11, filter_sigma=1.5,
    k1=0.01, k2=0.03, weights=None, data_format='NCHW'):
  # if img1.shape != img2.shape:
  #   raise RuntimeError('Input images must have the same shape (%s vs. %s).',
  #                      img1.shape, img2.shape)
  # if img1.ndim != 4:
  #   raise RuntimeError('Input images must have four dimensions, not %d',
  #                      img1.ndim)

  # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
  weights = weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
  levels = len(weights)
  weights = tf.constant(weights, dtype=tf.float32)
  im1, im2 = img1, img2
  n1, c1, h1, w1 = img1.shape
  n2, c2, h2, w2 = img2.shape
  mssim = None
  mcs = None
  for _ in range(levels):
    ssim, cs = tf_ssim(
        im1, im2, max_val=max_val, filter_size=filter_size,
        filter_sigma=filter_sigma, k1=k1, k2=k2, cs_map=True)
    ssim = tf.map_fn(tf.reduce_mean, ssim)
    cs = tf.map_fn(tf.reduce_mean, cs)
    ssim = ssim[:, None]
    cs = cs[:, None]
    mssim = ssim if mssim is None else tf.concat([mssim, ssim], axis=1)
    mcs = cs if mcs is None else tf.concat([mcs, cs], axis=1)
    im1 = downsample_2d(im1, data_format=data_format)
    im2 = downsample_2d(im2, data_format=data_format)
    h1 //= 2
    w1 //= 2
    h2 //= 2
    w2 //= 2
    im1.set_shape([n1, c1, h1, w1])
    im2.set_shape([n2, c2, h2, w2])
  a = mssim[:, -1] ** weights[-1]
  b = mcs[:, 0:-1] ** weights[0:-1]
  b = tf.reduce_prod(b, axis=1)
  return a * b

def _i(x): return tf.transpose(x, [0,2,3,1])
def _o(x): return tf.transpose(x, [0,3,1,2])


def dimensions(img):
  return img.shape.as_list()

def channels(img):
  return dimensions(img)[-1]

# https://stackoverflow.com/a/46991488/9919772
def rgb_yiq(rgb, axis=-1):
  p1r, p1g, p1b = tf.split(rgb, 3, axis=axis)
  y1y = (0.299*p1r + 0.587*p1g + 0.114*p1b)
  y1i = (0.596*p1r - 0.275*p1g - 0.321*p1b)
  y1q = (0.212*p1r - 0.523*p1g + 0.311*p1b)
  return y1y, y1i, y1q

def to_gray(img, c=None, axis=-1):
  if c is None:
    c = channels(img)
  if c == 3:
    y, i, q = rgb_yiq(img, axis=axis)
    return y
  else:
    assert c == 1
    return img

def tf_similarity(images, **kws):
  #n, c, h, w = images.shape.as_list()
  imgs1 = images
  imgs2 = tf.concat([imgs1[0:-1], [imgs1[-1]]], axis=0)
  imgs1 = to_gray(imgs1, c=3, axis=1)
  imgs2 = to_gray(imgs2, c=3, axis=1)
  #imgs2.set_shape([n, c, h, w])
  return tf_ssim_multiscale(imgs1, imgs2, **kws)
  # result = tf.image.ssim_multiscale(_i(imgs1), _i(imgs2), 1.0)
  # result = tf.stop_gradient(result)
  # return result
