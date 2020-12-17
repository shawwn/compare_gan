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

import gin
import os
import re


summary = tf.contrib.summary  # TensorFlow Summary API v2.

from tensorflow.python.ops.summary_ops_v2 import record_if


TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn countdown init")

@gin.configurable(blacklist=["log_dir"])
class TpuSummaries(object):
  """Class to simplify TF summaries on TPU.

  An instance of the class provides simple methods for writing summaries in the
  similar way to tf.summary. The difference is that each summary entry must
  provide a reduction function that is used to reduce the summary values from
  all the TPU cores.
  """

  def __init__(self, log_dir, save_summary_steps=1, save_image_steps=50, append_shapes=False):
    self._log_dir = log_dir
    self._image_entries = []
    self._scalar_entries = []
    # While False no summary entries will be added. On TPU we unroll the graph
    # and don't want to add multiple summaries per step.
    self.record = True
    self._save_summary_steps = save_summary_steps
    self._save_image_steps = save_image_steps
    self._append_shapes = append_shapes
    #assert TpuSummaries.inst is None
    TpuSummaries.inst = self

  def has(self, name):
    for entry in self._image_entries + self._scalar_entries:
      if entry.name == name:
        return True
    return False

  def image(self, name, tensor, reduce_fn):
    """Add a summary for images. Tensor must be of 4-D tensor."""
    if not self.record:
      return
    if self.has(name):
      logging.info("TpuSummaries.image: skipping duplicate %s", name)
    else:
      tensor = tf.convert_to_tensor(tensor)
      if self._append_shapes:
        name += '_' + '{}'.format(tensor.shape).strip('()').replace(', ', 'x')
      self._image_entries.append(
          TpuSummaryEntry(summary.image, name, tensor, reduce_fn, countdown=None, init=None))

  def scalar(self, name, tensor, reduce_fn=tf.math.reduce_mean, countdown=None, init=None):
    """Add a summary for a scalar tensor."""
    if not self.record:
      return
    # if we're sampling from the EMA model, don't graph any scalars.
    current_scope = gin.current_scope_str()
    if re.search(r"\bema\b", current_scope):
      logging.info("TpuSummaries.scalar: skipping EMA scalar %s", name)
      return
    if self.has(name):
      logging.info("TpuSummaries.scalar: skipping duplicate %s", name)
    else:
      tensor = tf.convert_to_tensor(tensor)
      if self._append_shapes:
        name += '_' + '{}'.format(tensor.shape).strip('()').replace(', ', 'x')
      if tensor.shape.ndims == 0:
        tensor = tf.expand_dims(tensor, 0)
      self._scalar_entries.append(
          TpuSummaryEntry(summary.scalar, name, tensor, reduce_fn, countdown, init))

  def get_host_call(self):
    """Returns the tuple (host_call_fn, host_call_args) for TPUEstimatorSpec."""
    # All host_call_args must be tensors with batch dimension.
    # All tensors are streamed to the host machine (mind the band width).
    global_step = tf.train.get_or_create_global_step()
    host_call_args = [tf.expand_dims(global_step, 0)]
    host_call_args.extend([e.tensor for e in self._image_entries])
    host_call_args.extend([e.tensor for e in self._scalar_entries])
    logging.info("host_call_args: %s", host_call_args)
    return (self._host_call_fn, host_call_args)

  def _host_call_fn(self, step, *args):
    """Function that will run on the host machine."""
    # Host call receives values from all tensor cores (concatenate on the
    # batch dimension). Step is the same for all cores.
    step = step[0]
    logging.info("host_call_fn: args=%s", args)
    with tf.name_scope('summary_writer/images'):
      with summary.create_file_writer(os.path.join(self._log_dir, 'images'), name='summary_writer_images').as_default():
        offset = 0
        with summary.record_summaries_every_n_global_steps(
                self._save_image_steps, step):
          for i, e in enumerate(self._image_entries):
            value = e.reduce_fn(args[i + offset])
            e.summary_fn(e.name, value, step=step)
        offset += len(self._image_entries)
    with tf.name_scope('summary_writer/scalars'):
      with summary.create_file_writer(os.path.join(self._log_dir, 'scalars'), name='summary_writer_scalars').as_default():
        with summary.record_summaries_every_n_global_steps(
              self._save_summary_steps, step):
          for i, e in enumerate(self._scalar_entries):
            value = e.reduce_fn(args[i + offset])
            ready = None
            if e.countdown is not None and False: # disable countdown for now
              with tf.device("cpu:0"):
                countdown = tf.get_local_variable(
                  e.name + "_countdown",
                  initializer=tf.constant_initializer(e.countdown),
                  shape=(),
                  dtype=tf.int64,
                  trainable=False,
                  use_resource=True)
                countdown_decrement = countdown.assign_sub(tf.sign(countdown))
                with tf.control_dependencies([countdown_decrement]):
                  ready = tf.less_equal(countdown, 0)
            if e.init is not None and False: # Disable this for now
              op = tf.reduce_any(tf.not_equal(value, tf.cast(e.init, value.dtype)))
              ready = tf.logical_and(ready, op) if ready is not None else op
            if ready is not None:
              with record_if(ready):
                e.summary_fn(e.name, value, step=step)
            else:
              e.summary_fn(e.name, value, step=step)
        offset += len(self._scalar_entries)
    ops = summary.all_summary_ops()
    logging.info("host_call: summary.all_summary_ops(): %s", ops)
    return ops



TpuSummaries.inst = None
