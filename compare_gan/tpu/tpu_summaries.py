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

"""Provide a helper class for using summaries on TPU via a host call.

TPUEstimator does not support writing TF summaries out of the box and TPUs can't
perform operations that write files to disk. To monitor tensor values during
training you can copy the tensors back to the CPU of the host machine via
a host call function. This small library provides a convienent API to do this.

Example:
from compare_gan.tpu import tpu_summaries
def model_fn(features, labels, params, mode):
  summary = tpu_summries.TpuSummaries(my_model_dir)

  summary.scalar("my_scalar_summary", tensor1)
  summary.scalar("my_counter", tensor2, reduce_fn=tf.math.reduce_sum)

  return TPUEstimatorSpec(
      host_call=summary.get_host_call(),
      ...)

Warning: The host call function will run every step. Writing large tensors to
summaries can slow down your training. High ranking outfeed operations in your
XProf profile can be an indication for this.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import tensorflow as tf

from compare_gan import tensorfork_tf as ttf

import gin
import os
import contextlib
import re
import time

summary = tf.contrib.summary  # TensorFlow Summary API v2.


TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn")

@gin.configurable(blacklist=["log_dir"])
class TpuSummaries(object):
  """Class to simplify TF summaries on TPU.

  An instance of the class provides simple methods for writing summaries in the
  similar way to tf.summary. The difference is that each summary entry must
  provide a reduction function that is used to reduce the summary values from
  all the TPU cores.
  """

  def __init__(self, log_dir, run_name='run', save_summary_steps=1, save_image_steps=50):
    assert re.match('^[a-z0-9]+[a-z-_0-9/]*$', run_name)
    assert '//' not in run_name
    assert not run_name.endswith('/')
    self._log_dir = log_dir
    self._run_name = run_name
    self._log_date = time.strftime('%Y-%m-%d-%H-%M-%S')
    self._image_entries = []
    self._scalar_entries = []
    # While False no summary entries will be added. On TPU we unroll the graph
    # and don't want to add multiple summaries per step.
    self.record = True
    self._save_summary_steps = save_summary_steps
    self._save_image_steps = save_image_steps
    #assert TpuSummaries.inst is None
    TpuSummaries.inst = self

  def has(self, name):
    name = ttf.variable_name(name)
    for entry in self._image_entries + self._scalar_entries:
      if entry.name == name:
        return True
    return False

  def get_var(self, name, *args, **kws):
    v = ttf.get_var(name, *args, **kws)
    self.scalar(os.path.join('knobs', v.name), v)
    return v

  def image(self, name, tensor, reduce_fn):
    """Add a summary for images. Tensor must be of 4-D tensor."""
    if not self.record:
      return
    name = ttf.variable_name(name)
    if self.has(name):
      logging.info("TpuSummaries.image: skipping duplicate %s", name)
    else:
      self._image_entries.append(
          TpuSummaryEntry(summary.image, name, tensor, reduce_fn))

  def scalar(self, name, tensor, reduce_fn=tf.math.reduce_mean):
    """Add a summary for a scalar tensor."""
    if not self.record:
      return
    name = ttf.variable_name(name)
    if self.has(name):
      logging.info("TpuSummaries.scalar: skipping duplicate %s", name)
    else:
      tensor = tf.convert_to_tensor(tensor)
      if tensor.shape.ndims == 0:
        tensor = tf.expand_dims(tensor, 0)
      self._scalar_entries.append(
          TpuSummaryEntry(summary.scalar, name, tensor, reduce_fn))

  def get_host_call(self):
    """Returns the tuple (host_call_fn, host_call_args) for TPUEstimatorSpec."""
    # All host_call_args must be tensors with batch dimension.
    # All tensors are streamed to the host machine (mind the band width).
    global_step = tf.train.get_or_create_global_step()
    host_call_args = [tf.expand_dims(global_step, 0)]
    host_call_args.extend([e.tensor for e in self._image_entries])
    host_call_args.extend([e.tensor for e in self._scalar_entries])
    logging.info("host_call_args: %r images and %r scalars", len(self._image_entries), len(self._scalar_entries))
    return (self._host_call_fn, host_call_args)

  def get_log_path(self, category):
    return os.path.join(self._log_dir, 'logs', category, self._run_name, category + '-' + self._log_date)

  @contextlib.contextmanager
  def log_every_n(self, category, n, current_step, ops):
    with summary.create_file_writer(self.get_log_path(category)).as_default():
      with summary.record_summaries_every_n_global_steps(tf.cast(n, dtype=tf.int64), current_step):
        yield
      ops.extend(summary.all_summary_ops())

  def _host_call_fn(self, step, *args):
    """Function that will run on the host machine."""
    # Host call receives values from all tensor cores (concatenate on the
    # batch dimension). Step is the same for all cores.
    logging.info("host_call_fn: len(args)=%r", len(args))
    step = step[0]
    save_image_steps = self.get_var("TpuSummaries.save_image_steps", self._save_image_steps)
    save_summary_steps = self.get_var("TpuSummaries.save_summary_steps", self._save_summary_steps)
    images = args[0:len(self._image_entries)]
    scalars = args[len(self._image_entries):]
    ops = []
    with self.log_every_n('images', save_image_steps, step, ops):
      for e, image in zip(self._image_entries, images):
        value = e.reduce_fn(image)
        e.summary_fn(e.name, value, step=step)
    with self.log_every_n('scalars', save_summary_steps, step, ops):
      for e, scalar in zip(self._scalar_entries, scalars):
        value = e.reduce_fn(scalar)
        e.summary_fn(e.name, value, step=step)
      # log extra global values.
      global_step = tf.train.get_or_create_global_step()
      summary.scalar('debug/step', step, step=step)
      summary.scalar('debug/global_step', global_step, step=global_step)
      summary.scalar('debug/global_step_minus_step', tf.identity(global_step - step), step=step)
    return tf.group(ops, name="host_call_log_ops")


TpuSummaries.inst = None
