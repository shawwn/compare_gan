
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint as pp
from pprint import pformat as pf
import contextlib
from contextlib import contextmanager

import sys
import os
import re
import six
import json
import base64
from six.moves.urllib.error import URLError

import tensorflow as tf
logging = tf.compat.v1.logging

def dotenv_reload(dotenv_path=None, dotenv_verbose=None):
  from dotenv import load_dotenv
  if dotenv_path is None:
    dotenv_path = os.environ.get('DOTENV_PATH')
  if dotenv_verbose is None:
    dotenv_verbose = bool(int(os.environ.get('DOTENV_VERBOSE', '1')))
  logging.info('load_dotenv(dotenv_path={!r}, verbose={!r})'.format(dotenv_path, dotenv_verbose))
  return load_dotenv(dotenv_path=dotenv_path, verbose=dotenv_verbose)

def dotenv_startup():
  if not bool(int(os.environ.get('NO_DOTENV', '0'))):
    dotenv_reload()
  else:
    logging.info('NO_DOTENV is set; not loading .env')

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python import framework
from tensorflow.python.client import session
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.compat.v1.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.python.eager.context import LogicalDevice
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.tpu import topology as topology_lib

import gin

try:
  from cloud_tpu_client import client  # pylint: disable=g-import-not-at-top
except ImportError:
  try:
    logging.debug(
        'Falling back to TensorFlow client; we recommended you install the Cloud '
        'TPU client directly with pip install cloud-tpu-client.')
    from tensorflow.python.tpu.client import client  # pylint: disable=g-import-not-at-top
  except ImportError:
    client = None

def _tpu_host():
  return os.environ.get('TPU_HOST')

def reroute(addr, host=None):
  if host is False:
    return addr
  if host is None:
    host = _tpu_host()
  if addr.startswith('grpc://'):
    return 'grpc://' + reroute(addr[len('grpc://'):], host=host)
  if not re.match('[0-9]+[.][0-9]+[.][0-9]+[.][0-9]+[:]8470', addr):
    return addr
  if not addr.endswith(':8470'):
    return addr
  a, b, c, d = [int(x) for x in addr.split(':')[0].split('.')]
  if a == 10 and b in [48, 49]:
    assert (d == 2)
    port = b * 1000 + c
  elif a == 10 and b in range(2, 66) and c == 0:
    port = b * 1000 + d
  else:
    return addr
  return host + ':' + str(port)

import functools
from collections import OrderedDict
from tensorflow.python.platform import test
mock = test.mock

import threading
from types import SimpleNamespace as NS

mocks = globals().get('mocks') or NS(advice={}, deactivate=None, lock=threading.RLock())

def mocks_active():
  return mocks.deactivate is not None

def mock_method(unique_name, cls, name=None, doc=None, create=False):
  def func(fn):
    nonlocal name
    if name is None:
      name = fn.__name__
    wrapped = getattr(cls, name, None)
    @functools.wraps(fn)
    def _fn(self, *args, **kwargs):
      return fn(wrapped, cls, self, *args, **kwargs)
    if hasattr(wrapped, '__name__'):
      _fn.__name__ = wrapped.__name__
    if hasattr(wrapped, '__module__'):
      _fn.__module__ = wrapped.__module__
    if hasattr(wrapped, '__qualname__'):
      _fn.__qualname__ = wrapped.__qualname__
    mocks.advice[unique_name] = lambda: mock.patch.object(cls, name, _fn, create=create)
    return _fn
  return func

def deactivate_mocks():
  with mocks.lock:
    if mocks.deactivate:
      mocks.deactivate()
      mocks.deactivate = None
      return True

def activate_mocks():
  with mocks.lock:
    deactivate_mocks()
    with contextlib.ExitStack() as stack:
      for creator in mocks.advice.values():
        stack.enter_context(creator())
      stk = stack.pop_all()
      mocks.deactivate = stk.close
      return stk

@mock_method('patch_resolver_auto_tpu', tpu_cluster_resolver.TPUClusterResolver, '__init__')
def resolver__init__(orig, cls, self, tpu=None, zone=None, project=None, *args, **kws):
  if tpu is None:
    tpu = os.environ.get('TPU_NAME')
  if zone is None:
    zone = os.environ.get('TPU_ZONE')
  if project is None:
    project = os.environ.get('TPU_PROJECT')
  return orig(self, tpu, zone, project, *args, **kws)

@mock_method('patch_resolver_master', tpu_cluster_resolver.TPUClusterResolver, 'master')
def _master(orig, cls, self, *args, **kws):
  ip = orig(self, *args, **kws)
  return reroute(ip)

@mock_method('patch_resolver_cluster_spec', tpu_cluster_resolver.TPUClusterResolver, 'cluster_spec')
def _cluster_spec(orig, cls, self, *args, **kws):
  spec = orig(self, *args, **kws)
  r = dict()
  for k, v in spec.as_dict().items():
    r[k] = [reroute(ip) for ip in v]
  return server_lib.ClusterSpec(r)

@mock_method('patch_fetch_cloud_tpu_metadata', (client.Client if client is not None else tpu_cluster_resolver.TPUClusterResolver), '_fetch_cloud_tpu_metadata')
def _fetch_cloud_tpu_metadata(orig, cls, self, *args, **kws):
  while True:
    try:
      return orig(self, *args, **kws)
    except Exception as e:
      if '[Errno 111] Connection refused' in str(e):
        # retry
        import time
        time.sleep(1.0)
      else:
        raise e

@mock_method('patch__parse_topology', topology_lib.Topology, '_parse_topology')
def _parse_topology(orig, cls, self, serialized=None, mesh_shape=None, device_coordinates=None):
  """Parses a serialized `TopologyProto` into `self`."""
  proto = topology_pb2.TopologyProto()
  proto.ParseFromString(serialized)

  self._mesh_shape = np.array(proto.mesh_shape, dtype=np.int32)
  if len(self._mesh_shape) not in [3, 4] or any(self._mesh_shape < 1):
    raise ValueError("`mesh_shape` must be a vector of size 3 or 4 with positive "
                     "entries; got {}".format(self._mesh_shape))

  if proto.num_tasks < 0:
    raise ValueError("`num_tasks` must be >= 0; got {}".format(
        proto.num_tasks))
  if proto.num_tpu_devices_per_task < 0:
    raise ValueError("`num_tpu_devices_per_task` must be >= 0; got {}".format(
        proto.num_tpu_devices_per_task))

  expected_coordinates_size = (proto.num_tasks * proto.num_tpu_devices_per_task * len( proto.mesh_shape))
  if len(proto.device_coordinates) != expected_coordinates_size:
    raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                     "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                     "got shape {}".format(proto.num_tasks,
                                           proto.num_tpu_devices_per_task,
                                           proto.mesh_shape,
                                           len(proto.device_coordinates)))

  coords = np.array(proto.device_coordinates, dtype=np.int32)
  if any(coords < 0):
    raise ValueError("`device_coordinates` must be >= 0")
  coords = coords.reshape((proto.num_tasks, proto.num_tpu_devices_per_task, len(proto.mesh_shape)))
  self._device_coordinates = coords
  expected_coordinates_size = (proto.num_tasks * proto.num_tpu_devices_per_task * len(proto.mesh_shape))
  if len(proto.device_coordinates) != expected_coordinates_size:
    raise ValueError("`device_coordinates` must have shape num_tasks ({}) * "
                     "num_tpu_devices_per_task ({}) * len(mesh_shape) ({}); "
                     "got shape {}".format(proto.num_tasks,
                                           proto.num_tpu_devices_per_task,
                                           proto.mesh_shape,
                                           len(proto.device_coordinates)))

@mock_method('patch__invert_topology', topology_lib.Topology, '_invert_topology')
def _invert_topology(orig, cls, self):
  """Inverts a [task,device,axis] topology to [x,y,z] -> task/device maps."""
  tasks = np.full(list(self.mesh_shape), -1, dtype=np.int32)
  devices = np.full(list(self.mesh_shape), -1, dtype=np.int32)
  if len(self.mesh_shape) == 3:
    for task in range(self.device_coordinates.shape[0]):
      for device in range(self.device_coordinates.shape[1]):
        x, y, z = self.device_coordinates[task, device, :]
        tasks[x, y, z] = task
        devices[x, y, z] = device
  else:
    for task in range(self.device_coordinates.shape[0]):
      for device in range(self.device_coordinates.shape[1]):
        x, y, z, core = self.device_coordinates[task, device, :]
        tasks[x, y, z, core] = task
        devices[x, y, z, core] = device
  return tasks, devices

from tensorflow.python.tpu import device_assignment as device_assignment_lib

# @mock_method('patch__device_assignment', device_assignment_lib, 'device_assignment')
# def _device_assignment(orig, cls, topology, computation_shape=None, computation_stride=None, num_replicas=1, **kws):
#   print('TKTK')
#   value = orig(topology, computation_shape=computation_shape, computation_stride=computation_stride, num_replicas=num_replicas, **kws)
#   return value


@contextmanager
def patch_tensorflow():
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity('DEBUG')
  tf.compat.v1.enable_resource_variables()
  dotenv_startup()
  gin.enter_interactive_mode()
  with activate_mocks():
    result = yield
    return result

def patch_tensorflow_interactive():
  patch = patch_tensorflow()
  patch.__enter__()
  return patch


def interact():
    import code
    code.InteractiveConsole(locals=globals()).interact()


from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

def make_session_config():
  #session_config = config_pb2.ConfigProto(allow_soft_placement=True, isolate_session_state=True)
  rpc_options = config_pb2.RPCOptions()
  # Setting cache_rpc_response to true will enable sender side caching of
  # response for RecvTensorAsync and RecvBufAsync to allow receiver to retry
  # requests . This is only necessary when the network fabric is experiencing a
  # significant error rate.  Without it we'll fail a step on an network error,
  # while with it we'll be able to complete long steps (like complex
  # initializations) in the face of some network errors during RecvTensor.
  rpc_options.cache_rpc_response = True
  rewriter_config = rewriter_config_pb2.RewriterConfig(
    disable_model_pruning=True,
    disable_meta_optimizer=True,
    dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF,
    fail_on_optimizer_errors=True,
    )
  graph_options = config_pb2.GraphOptions(
    rewrite_options=rewriter_config,
    place_pruned_graph=True,
    infer_shapes=True,
    )
  session_config = config_pb2.ConfigProto(
    graph_options=graph_options,
    allow_soft_placement=True,
    isolate_session_state=False,
    )
  # share variables across sessions on TPUs
  session_config.experimental.share_session_state_in_clusterspec_propagation = True
  # TODO: research this. What does it do?
  # session_config.share_cluster_devices_in_session = True
  return session_config


def get_resolver(tpu=None, zone=None, project=None):
  if tpu is None:
    tpu = os.environ.get('TPU_NAME')
  if zone is None:
    zone = os.environ.get('TPU_ZONE')
  if project is None:
    project = os.environ.get('TPU_PROJECT')
  try:
    return TPUClusterResolver(tpu=tpu, zone=zone, project=project)
  except ValueError:
    pass

from tensorflow.python.training import monitored_session

@mock_method('patch__monitored_session__raw_session', monitored_session._MonitoredSession, 'raw_session', create=True)
def MonitoredSession_raw_session(orig, cls, self):
  """Returns underlying `TensorFlow.Session` object."""
  return self._tf_sess()

@mock_method('patch__monitored_session__list_devices', monitored_session._MonitoredSession, 'list_devices', create=True)
def MonitoredSession_list_devices(orig, cls, self):
  return self._tf_sess().list_devices()


def get_session(graph=None, resolver=None, config=None, interactive=False, monitored=False, hooks=None):
  if monitored:
    return monitored_session.MonitoredSession(SessionCreator(graph=graph, resolver=resolver, config=config, interactive=interactive), hooks=hooks)
  if hooks is not None:
    raise ValueError('When monitored=False, hooks must be None')
  if graph is None:
    graph = tf.compat.v1.get_default_graph()
  if resolver is None:
    resolver = get_resolver()
  Session = tf.compat.v1.InteractiveSession if interactive else tf.compat.v1.Session
  master = resolver.master() if resolver is not None else None
  cluster_spec = resolver.cluster_spec() if resolver is not None else None
  config = get_session_config(cluster_spec=cluster_spec) if config is None else config
  return Session(master, graph=graph, config=config)

class SessionCreator:
  def __init__(self, graph=None, resolver=None, config=None, interactive=False):
    self.graph = graph
    self.resolver = resolver
    self.config = config
    self.interactive = interactive
  def create_session(self):
    return get_session(graph=self.graph, resolver=self.resolver, config=self.config, interactive=self.interactive)

def get_session_config(cluster_spec=None, config=None):
  if config is None:
    config = make_session_config()
  if cluster_spec is not None:
    cluster_def = cluster_spec.as_cluster_def()
    config.cluster_def.CopyFrom(cluster_def)
  return config

def clone_session(session=None, graph=None, interactive=False, **kws):
  if session is None:
    session = tf.compat.v1.get_default_session()
  if hasattr(session, 'raw_session'):
    session = session.raw_session()
  if graph is None:
    graph = session.graph
  config = session._config # is there a better way to do this?
  master = session.sess_str # is there a better way to do this?
  Session = (tf.compat.v1.InteractiveSession if interactive else tf.compat.v1.Session)
  return Session(master, graph=graph, config=config, **kws)

def reset_session(session=None, graph=None, interactive=True, **kws):
  if session is None:
    session = tf.compat.v1.get_default_session()
  if graph is None:
    graph = tf.Graph()
  graph.as_default().__enter__()
  session2 = clone_session(session, graph=graph, interactive=interactive, **kws)
  session2.as_default().__enter__()
  if 'sess' in globals():
    globals()['sess'] = session2
  return session2

from tensorflow.python.distribute import values

def enclosing_tpu_context():
  return values._enclosing_tpu_context()


from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_gradients
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


topology_cache = {}
try:
  with open('topology.cache', 'r') as f: topology_cache = json.load(f)
except FileNotFoundError:
  pass

def get_tpu_resolver(tpu_or_resolver=None, zone=None, project=None):
  if tpu_or_resolver is None or isinstance(tpu_or_resolver, str):
    tpu_or_resolver = TPUClusterResolver(tpu=tpu_or_resolver, zone=zone, project=project)
  return tpu_or_resolver

def get_tpu_name(tpu_or_resolver=None, zone=None, project=None):
  if isinstance(tpu_or_resolver, str):
    return tpu_or_resolver
  if tpu_or_resolver is None:
    try:
      tpu_or_resolver = TPUClusterResolver(tpu=tpu_or_resolver, zone=zone, project=project)
    except ValueError as e:
      if str(e) == 'Please provide a TPU Name to connect to.':
        return None
  name = tpu_or_resolver
  if hasattr(tpu_or_resolver, '_tpu'):
    name = tpu_or_resolver._tpu
  if isinstance(name, bytes):
    name = name.decode('utf8')
  return name

def cached_topology(tpu=None, zone=None, project=None):
  tpu = get_tpu_name(tpu, zone=zone, project=project)
  result = topology_cache.get(tpu, None)
  if result is not None:
    serialized = base64.b64decode(result)
    return topology_lib.Topology(serialized=serialized)

def get_topology(tpu=None, zone=None, project=None, force=False):
  tpu_topology = cached_topology(tpu=tpu, zone=zone, project=project)
  if tpu_topology is None or force:
    res = get_tpu_resolver(tpu, zone=zone, project=project)
    tpu_topology = tpu_strategy_util.initialize_tpu_system(res)
    tpu_name = get_tpu_name(res)
    topology_cache.update({tpu_name: base64.b64encode(tpu_topology.serialized()).decode('utf8')})
    topology_cache_contents = json.dumps(topology_cache, indent=4, sort_keys=True)
    with open('topology.cache', 'w') as f:
      f.write(topology_cache_contents)
  return tpu_topology

def get_tpu_total_core_count(topology=None):
  if topology is None:
    topology = cached_topology()
  return topology.num_tasks * topology.num_tpus_per_task

def get_tpu_cores(core_ids=None, topology=None):
  if topology is None:
    topology = cached_topology()
  if topology is None:
    return []
  all_cores = topology.device_coordinates.reshape([-1, topology.device_coordinates.shape[-1]])
  if core_ids is not None:
    coords = []
    for task_idx, task in enumerate(topology.device_coordinates):
      for core_idx, core in enumerate(task):
        core_id = (core_idx + task_idx*len(topology.device_coordinates[task_idx]))
        if core_id in core_ids:
          coords.append(core)
    return coords
    # cores = [[core for core_idx, core in enumerate(task) if (core_idx + task_idx*len(topology.device_coordinates[task_idx])) in core_ids] for task_idx, task in enumerate(topology.device_coordinates)]
    # #core_ids = np.array(cores, dtype=np.int32)
    # return all_cores[cores]
  return all_cores

from tensorflow.python.tpu import device_assignment as device_assignment_lib

def get_task_and_cores_to_replicas(topology=None):
  if topology is None:
    topology = cached_topology()
  return device_assignment_lib._compute_task_and_cores_to_replicas(topology.device_coordinates, topology)

def get_core_assignment(core_ids=None, topology=None):
  if topology is None:
    topology = cached_topology()
  return device_assignment_lib.DeviceAssignment(topology, [[topology.device_coordinates[i//8][i%8]] for i in core_ids])

def get_device_assignment(computation_shape=None, computation_stride=None, *, num_replicas=None, topology=None):
  if topology is None:
    topology = cached_topology()
  if num_replicas is None:
    dev = None
    core_count = get_tpu_total_core_count(topology=topology)
    for i in range(core_count):
      try:
        dev = get_device_assignment(computation_shape=computation_shape, computation_stride=computation_stride, num_replicas=i+1, topology=topology)
        num_replicas = i+1
      except ValueError:
        if dev is None:
          raise
        return dev
  device_assignment = tf.tpu.experimental.DeviceAssignment.build(topology, computation_shape=computation_shape, computation_stride=computation_stride, num_replicas=num_replicas)
  return device_assignment

def print_device_assignment(device_assignment):
  [print('--- replica %d ---' % i) or [
    print({
      'coordinate': device_assignment.coordinates(i,j),
      'host': device_assignment.host_device(i,j),
      'core': device_assignment.tpu_device(i,j),
      'ordinal': device_assignment.tpu_ordinal(i,j)}) for j in range(device_assignment.num_cores_per_replica)] for i in range(device_assignment.num_replicas)]
  print('=== num_replicas=%d num_cores_per_replica=%d ===' % (device_assignment.num_replicas, device_assignment.num_cores_per_replica))
  return device_assignment

from google.protobuf.json_format import MessageToJson
import json

def pb_to_json(pb):
  return json.loads(MessageToJson(pb))


import sys
import PIL.Image
from io import BytesIO
import os
import base64

def imgnorm(img):
  if img.dtype == 'float32':
    # Assumes range [-1 .. 1]
    img = img.copy()
    img *= 0.5
    img += 0.5
    img *= 255
    img = img.astype(np.uint8)
  return img

def image_to_bytes(img, format='PNG', **kwargs):
  b = BytesIO()
  img = numpy_to_image(img)
  img.save(b, format=format, **kwargs)
  return b.getvalue()

def osc():
  if os.environ.get('TERM').startswith('screen'):
    return "\033Ptmux;\033\033]"
  else:
    return "\033]"

def st():
  if os.environ.get('TERM').startswith('screen'):
    return "\a\033\\"
  else:
    return "\a"

def imgcat(img, name='test.png', width='auto', height='auto', preserve_aspect_ratio=True, *, file=sys.stdout):
  b = base64.b64encode(image_to_bytes(img)).decode('utf8')
  s = []
  s += [osc()]
  s += [('1337;File=')]
  s += ['name=%s;width=%s;height=%s;preserveAspectRatio=%s;' %
      (name, str(width), str(height), '1' if bool(int(preserve_aspect_ratio)) else '0')]
  s += ['size=%d' % len(b)]
  s += [';inline=%s' % '1']
  s += [':']
  s += ['%s' % b]
  s += ['%s' % st()]
  s = ''.join(s)
  file.write(s)

def numpy_to_image(img):
  if isinstance(img, PIL.Image.Image):
    return img
  return PIL.Image.fromarray(imgnorm(img))



from tensorflow.python.tpu import tpu as tpu_ops

def tpu_shard(op, device_assignment=None, num_shards=None, outputs_from_all_shards=True, topology=None, **kws):
  if topology is None:
    topology = cached_topology()
  if device_assignment is None:
    device_assignment = get_device_assignment(topology=topology)
  assert device_assignment is not None
  if num_shards is None:
    num_shards = len(device_assignment.core_assignment)
  return tpu_ops.shard(op, outputs_from_all_shards=outputs_from_all_shards, num_shards=num_shards, device_assignment=device_assignment, **kws)

if __name__ == '__main__':
  _tf_patch = patch_tensorflow_interactive()
  if len(sys.argv) <= 1 or bool(int(os.environ.get('REPL', '0'))) or os.environ.get('TPU_NAME'):
    tf1 = tf.compat.v1

    session_config = None
    
    master = None
    res = None
    cluster_spec = None
    cluster_def = None
    job_names = None
    master_job = 'worker'
    try:
      if 'TPU_NAME' in os.environ:
        res = TPUClusterResolver()
        master = res.get_master()
        cluster_spec = res.cluster_spec()
      elif 'TPU_IP' in os.environ:
        master = os.environ['TPU_IP'].replace('grpc://', '')
        if ':' not in master:
          master = master + ':8470'
        master = reroute('grpc://' + master)
      session_config = get_session_config(cluster_spec=cluster_spec)
      if cluster_spec is not None:
        cluster_def = cluster_spec.as_cluster_def()
        job_names = set([job.name for job in cluster_def.job])
        assert len(job_names) == 1
        master_job = cluster_def.job[0].name
    except:
      import traceback
      traceback.print_exc()
    #graph = tf.Graph()
    graph = tf.compat.v1.get_default_graph()
    #sess = tf.compat.v1.InteractiveSession(master, graph=graph, config=session_config)
    sess = get_session(config=session_config, monitored=True, interactive=True)
    devices = sess.list_devices()
    cores = sorted([x.name for x in devices if ':TPU:' in x.name])
    num_cores = len(cores)
    print(cluster_def)
    print('cores: %d ip: %s' % (num_cores, master))
    r = sess.run
    from importlib import reload
    from tensorflow.python.tpu import tpu as tpu_ops
    from tensorflow.compiler.tf2xla.python import xla
    from tensorflow.compiler.tf2xla.ops import gen_xla_ops
    from tensorflow.python.tpu import tpu_strategy_util
    from tensorflow.python.tpu import topology as topology_lib
    tpu_topology = None
    if num_cores > 0:
      tpu_topology = cached_topology()

  if len(sys.argv) > 1:
    filename = sys.argv[1]
    sys.argv = sys.argv[1:]
    with open(filename) as f:
      source = f.read()
    code = compile(source, filename, 'exec')
    exec(code, globals(), locals())

