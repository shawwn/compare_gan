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

"""Tests custom TensorFlow operations for TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import parameterized
from compare_gan.tpu import tpu_ops
import numpy as np
import tensorflow as tf
import os
from tensorflow.compat.v1.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.tpu import tpu_strategy_util
from tensorflow.python.tpu import device_assignment as device_assignment_lib

import base64


class TpuOpsTpuTest(parameterized.TestCase, tf.test.TestCase):

  def get_resolver(self):
    if '_resolver' not in globals():
      globals()['_resolver'] = TPUClusterResolver(os.environ['TPU_NAME'])
    return globals()['_resolver']

  def cached_topology(self):
    result = os.environ.get('TPU_TOPOLOGY', None)
    if result is not None:
      serialized = base64.b64decode(result)
      return topology_lib.Topology(serialized=serialized)

  def get_topology(self):
    if '_topology' not in globals():
      topology = self.cached_topology()
      if topology is None:
        topology = tpu_strategy_util.initialize_tpu_system(self.get_resolver())
        print(base64.b64encode(globals()['_topology'].serialized()).decode('utf8'))
      globals()['_topology'] = topology
    return globals()['_topology']

  def get_core_assignment(self, *core_ids):
    return device_assignment_lib.DeviceAssignment(self.get_topology(), [[self.get_topology().device_coordinates[0][i]] for i in core_ids])

  def get_session(self, graph=None):
    if graph is None:
      graph = tf.compat.v1.get_default_graph()
    session_config = config_pb2.ConfigProto(
        allow_soft_placement=True,
        isolate_session_state=False,
        )
    res = self.get_resolver()
    master = res.master()
    cluster_spec = res.cluster_spec()
    if cluster_spec:
      cluster_def = cluster_spec.as_cluster_def()
      session_config.cluster_def.CopyFrom(cluster_def)
    sess = tf.compat.v1.Session(master, graph=graph, config=session_config)
    return sess

  @property
  def num_shards(self):
    return len([dev for dev in self.get_session().list_devices() if ':TPU:' in dev.name])


  #def testRunsOnTpu(self):
  #  """Verify that the test cases runs on a TPU chip and has 2 cores."""
  #  expected_device_names = [
  #      "/job:localhost/replica:0/task:0/device:CPU:0",
  #      "/job:localhost/replica:0/task:0/device:TPU:0",
  #      "/job:localhost/replica:0/task:0/device:TPU:1",
  #      "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0",
  #  ]
  #  with self.get_session() as sess:
  #    devices = sess.list_devices()
  #    logging.info("devices:\n%s", "\n".join([str(d) for d in devices]))
  #    self.assertAllEqual([d.name for d in devices], expected_device_names)

  def testCrossReplicaConcat(self):
    def computation(x, replica_id):
      logging.info("x: %s\nreplica_id: %s", x, replica_id[0])
      return tpu_ops.cross_replica_concat(x, replica_id[0], num_replicas=2)

    inputs = np.asarray([[3, 4], [1, 5]])
    expected_output = np.asarray([[3, 4], [1, 5], [3, 4], [1, 5]])

    with tf.Graph().as_default():
      x = tf.constant(inputs)
      replica_ids = tf.constant([0, 1], dtype=tf.int32)
      x_concat, = tf.contrib.tpu.batch_parallel(
          computation, [x, replica_ids], num_shards=2, device_assignment=self.get_core_assignment(0, 1))
      self.assertAllEqual(x.shape.as_list(), [2, 2])
      self.assertAllEqual(x_concat.shape.as_list(), [4, 2])

      with self.get_session() as sess:
        #sess.run(tf.contrib.tpu.initialize_system())
        sess.run(tf.global_variables_initializer())
        x_concat = sess.run(x_concat)
        logging.info("x_concat: %s", x_concat)
        self.assertAllClose(x_concat, expected_output)

  # Test with group size 2 (test case has 2 cores, so this global batch norm).
  @parameterized.parameters(
      {"group_size": None},  # Defaults to number of TPU cores.
      {"group_size": 0},  # Defaults to number of TPU cores.
      {"group_size": 2},
  )
  def testCrossReplicaMean(self, group_size):
    # Verify that we average across replicas by feeding 2 vectors to the system.
    # Each replica should get one vector which is then averaged across
    # all replicas and simply returned.
    # After that each replica has the same vector and since the outputs gets
    # concatenated we see the same vector twice.
    inputs = np.asarray(
        [[0.55, 0.70, -1.29, 0.502], [0.57, 0.90, 1.290, 0.202]],
        dtype=np.float32)
    expected_output = np.asarray(
        [[0.56, 0.8, 0.0, 0.352], [0.56, 0.8, 0.0, 0.352]], dtype=np.float32)

    def computation(x):
      self.assertAllEqual(x.shape.as_list(), [1, 4])
      return tpu_ops.cross_replica_mean(x, group_size=group_size)

    with tf.Graph().as_default():
      # Note: Using placeholders for feeding TPUs is discouraged but fine for
      # a simple test case.
      x = tf.compat.v1.placeholder(name="x", dtype=tf.float32, shape=inputs.shape)
      y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2, device_assignment=self.get_core_assignment(0, 1))
      with self.get_session() as sess:
        #sess.run(tf.contrib.tpu.initialize_system())
        # y is actually a list with one tensor. computation would be allowed
        # to return multiple tensors (and ops).
        actual_output = sess.run(y, {x: inputs})[0]

    self.assertAllEqual(actual_output.shape, (2, 4))
    self.assertAllClose(actual_output, expected_output)

  def testCrossReplicaMeanGroupSizeOne(self, group_size=1):
    # Since the group size is 1 we only average over 1 replica.
    inputs = np.asarray(
        [[0.55, 0.70, -1.29, 0.502], [0.57, 0.90, 1.290, 0.202]],
        dtype=np.float32)
    expected_output = np.asarray(
        [[0.55, 0.7, -1.29, 0.502], [0.57, 0.9, 1.290, 0.202]],
        dtype=np.float32)

    def computation(x):
      self.assertAllEqual(x.shape.as_list(), [1, 4])
      return tpu_ops.cross_replica_mean(x, group_size=group_size)

    with tf.Graph().as_default():
      # Note: Using placeholders for feeding TPUs is discouraged but fine for
      # a simple test case.
      x = tf.compat.v1.placeholder(name="x", dtype=tf.float32, shape=inputs.shape)
      y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2, device_assignment=self.get_core_assignment(0, 1))
      with self.get_session() as sess:
        #sess.run(tf.contrib.tpu.initialize_system())
        # y is actually a list with one tensor. computation would be allowed
        # to return multiple tensors (and ops).
        actual_output = sess.run(y, {x: inputs})[0]

    self.assertAllEqual(actual_output.shape, (2, 4))
    self.assertAllClose(actual_output, expected_output)

  def parallel(self, computation, group_size):
      y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2, device_assignment=self.get_core_assignment(0, 1))
      return y

  # Test with group size 2 (test case has 2 cores, so this global batch norm).
  @parameterized.parameters(
      {"group_size": None},  # Defaults to number of TPU cores.
      {"group_size": 0},  # Defaults to number of TPU cores.
      {"group_size": 1},
      {"group_size": 2},
      {"group_size": 4},
      {"group_size": 8},
  )
  def testCrossReplicaMoments(self, group_size):
    num_shards = self.num_shards
    gs = (group_size or num_shards)
    print('group_size', group_size)
    print('gs', gs)
    inputs = np.random.RandomState(seed=0).uniform(np.zeros([num_shards * 8, 4], dtype=np.float32))
    for i in range(0, len(inputs), 2): inputs[i] = 0.0
    expected_mean = inputs.mean(0)
    expected_var = inputs.var(0)

    def computation(x):
      print(x, 'x')
      #self.assertAllEqual(x.shape.as_list(), [num_shards * 100 // gs, 4])
      #self.assertAllEqual(x.shape.as_list(), [len(inputs) // num_shards, 4])
      mean, var = tpu_ops.cross_replica_moments(x, axis=(0,), group_size=group_size)
      print(mean, 'mean')
      print(var, 'var')
      return tf.reshape(mean, [-1, 4]), tf.reshape(var, [-1, 4])

    with tf.Graph().as_default():
      # Note: Using placeholders for feeding TPUs is discouraged but fine for
      # a simple test case.
      x = tf.compat.v1.placeholder(name="x", dtype=tf.float32, shape=inputs.shape)
      #y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2, device_assignment=self.get_core_assignment(0, 1))
      if group_size == 2:
        y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=2, device_assignment=self.get_core_assignment(0, 1))
      else:
        y = tf.contrib.tpu.batch_parallel(computation, inputs=[x], num_shards=8, device_assignment=self.get_core_assignment(0, 1, 2, 3, 4, 5, 6, 7))
      with self.get_session() as sess:
        #sess.run(tf.contrib.tpu.initialize_system())
        # y is actually a list with one tensor. computation would be allowed
        # to return multiple tensors (and ops).
        actual_mean, actual_var = sess.run(y, {x: inputs})
        logging.info("expected_mean:    %s %s", expected_mean.shape, expected_mean)
        logging.info("actual_mean[0]: %s %s", actual_mean.shape, actual_mean[0])
        logging.info("actual_mean:    %s %s %s", actual_mean.shape, actual_mean.mean(0), actual_mean)
        logging.info("expected_var:    %s %s", expected_var.shape, expected_var)
        logging.info("actual_var[0]: %s %s", actual_var.shape, actual_var[0])
        logging.info("actual_var:    %s %s %s", actual_var.shape, actual_var.mean(0), actual_var)
        #import pdb; pdb.set_trace()

    #self.assertAllEqual(actual_mean.shape, (2, 4))
    self.assertAllClose(actual_mean.mean(0), expected_mean)
    self.assertAllClose(actual_mean[0], expected_mean)
    #self.assertAllEqual(actual_mean.shape, (2, 4))
    self.assertAllClose(actual_var.mean(0), expected_var)
    self.assertAllClose(actual_var[0], expected_var)


if __name__ == "__main__":
  tf.test.main()
