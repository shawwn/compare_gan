from . import tensorfork
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import importlib
import threading
import gin
import gin.tf.external_configurables

from absl import logging

import traceback
import importlib

class Context:
  def __init__(self):
    pass

if 'api' not in globals():
  api = Context()

if 'vals' not in globals():
  vals = Context()

def register_global(name, x):
  assert not hasattr(vals, name)
  setattr(vals, name, x)

class State:
  def __init__(self):
    pass

if 'state' not in globals():
  state = State()
  state.pinned_sessions = []
  state.session_timeout_in_ms = 1200000
  state.eval_lightweight_timeout = 20000
  state.load_lightweight_timeout = 20000
  #state.initialize_timeout = 60000
  state.initialize_timeout = 20*60000
  state.context_load_timeout = 20000
  state.ensure_on_init = True
  state.release_trainer_sema = True
  state.tpu_init_timeout = 10000
  state.summary_log_timeout = 20000
  state.use_global_data_sampler = True
  state.shuffle_cycles = True
  state.vars = {}

def api_op(f):
  if hasattr(api, f.__name__):
    logging.warning("Redefining %s", f.__name__)
  setattr(api, f.__name__, f)
  module = importlib.__import__('compare_gan.tensorfork_tf')
  setattr(module, f.__name__, f)
  #module.__globals__[f.__name__] = f
  globals()[f.__name__] = f
  return f

api_op = api_op(api_op)

@api_op
def eval_lightweight(variable, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return None
  if timeout_in_ms is None:
    timeout_in_ms = state.eval_lightweight_timeout
  return api.eval(variable, session=session, timeout_in_ms=timeout_in_ms)

@api_op
def load_lightweight(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return
  if timeout_in_ms is None:
    timeout_in_ms = state.load_lightweight_timeout
  return api.load(variable, value, session=session, timeout_in_ms=timeout_in_ms)

from tensorflow.core.protobuf import config_pb2

@api_op
def load(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return
  ops = variable.initializer
  vals = dict([(variable.initializer.inputs[1], value)])
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(ops, vals, options=options)

@api_op
def eval(variable, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return None
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(variable, options=options)

@api_op
def run(ops, timeout=10.0, session=None, **kws):
  if not isinstance(ops, list):
    ops = [ops]
  session = session or tf.get_default_session()
  if session is None:
    logging.warning('Session is None')
    return None
  assert 'options' not in kws
  options = None
  if timeout is not None:
    options=config_pb2.RunOptions(timeout_in_ms=int(timeout*1000))
  return session.run(ops, options=options, **kws)

@api_op
def run2(ops, timeout=20.0, session=None, **kws):
  return api.run(ops, timeout=timeout, session=session, **kws)

import re

@api_op
def variable_name(variable):
  if hasattr(variable, 'name'):
    variable = variable.name
  if re.match(r'core[0-9]+/', variable):
    variable = variable.split('/', 1)[-1]
  variable = variable.split(':', 1)[0]
  return variable

@api_op
def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = api.variable_name(variable).split(':')[0]
    value = reader.get_tensor(name)
    value = api.truncate_value(variable, value, reshape=reshape)
    yield variable, value

import numpy as np
import math
import sys

@api_op
def truncate_value(variable, value, reshape=True):
  if not reshape:
    return value
  shape = variable.shape.as_list()
  params = np.prod(shape)
  params2 = np.prod(value.shape)
  if params == params2:
    return value
  if params2 > params:
    logging.info('Truncating %s from shape %s to shape %s. var=%s', variable.name, value.shape, shape, variable)
    sys.stdout.flush()
    value = np.array(value)
    value = value.reshape([-1])
    value = value[0:params]
    value = value.reshape(shape)
  else:
    logging.info('Expanding %s from shape %s to shape %s. var=%s', variable.name, value.shape, shape, variable)
    sys.stdout.flush()
    value = np.array(value)
    value = value.reshape([-1])
    n = math.ceil(params / params2)
    value = np.tile(value, n)
    value = value.reshape(shape)
  return value

@api_op
def assign_values(variables, values, session=None, timeout_in_ms=60000):
  session = session or tf.get_default_session()
  ops = [x.initializer for x in variables]
  vals = dict([(x.initializer.inputs[1], value) for x, value in zip(variables, values)]) # TODO: bfloat16 support
  #for x, (k, v) in zip(variables, vals.items()):
  #  print(x.name, x.shape.as_list(), k, v.shape)
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  session.run(ops, vals, options=options)

import os
import re

@api_op
def fqn(name, scope=None):
  if scope is None or len(scope) <= 0:
    scope = tf.get_variable_scope().name
  name = '_'.join(re.split('[^a-z0-9]', os.path.join(scope, name)))
  return name

@api_op
def absolute_name_scope(scope):
  return tf.name_scope(scope + "/")

@api_op
def absolute_variable_scope(scope=None, **kwargs):
  if scope is None:
    scope = tf.get_variable_scope().name
  return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)

@api_op
def scoped_name(name=None, scope=None):
  if scope is None:
    scope = tf.get_variable_scope().name
  if name is None:
    name = ''
  name = name.split(':')[0]
  return os.path.join(scope, name)

@api_op
def get_var(name, default_value=None, update=False, scope=None, dtype=None, shape=(), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES], use_resource=True, **kws):
  if scope is not None:
    try:
      return api.get_var(os.path.join(scope, name), default_value=None, update=update, scope=None, dtype=dtype, shape=shape, trainable=trainable, collections=collections, use_resource=use_resource, **kws)
    except ValueError:
      scope = None
  knob = name if '.' in name else fqn(name)
  #knob = knob.split('/')[-1]
  if not tensorfork.has_knob(knob):
    update = True
  if update and default_value is None:
    value = tensorfork.knobs(knob)
    if value is None:
      raise ValueError("No such knob: {}".format(knob))
  else:
    value = tensorfork.knobs(knob, default_value, update)
  if dtype is None:
    if hasattr(value, 'dtype'):
      dtype = value.dtype
    elif isinstance(value, float):
      dtype = tf.float32
    elif isinstance(value, int):
      dtype = tf.int32
    else:
      raise ValueError("Unknown type")
  with absolute_variable_scope(reuse=tf.AUTO_REUSE):
    initializer = kws.pop('initializer', None)
    if initializer is None:
      initializer = tf.constant_initializer(value) if value is not None else None
    logging.info("tf.get_variable(name=%s, shape=%s, initializer=%s, default_value=%s, value=%s)", name, shape, initializer, default_value, value)
    var = tf.get_variable(name=name, dtype=dtype, shape=shape, trainable=trainable, collections=collections, use_resource=use_resource, initializer=initializer, **kws)
  state.vars[variable_name(var.name)] = {'knob': knob, 'variable': var}
  if update or default_value is None:
    api.load_lightweight(var, value)
  v = api.eval_lightweight(var)
  if v is not None and v != value:
    tensorfork.knobs(name, v, update=True)
  return var

@api_op
def compare_values(a, b, eps=1e-10):
  if abs(a - b) < eps:
    return 0
  if a > b:
    return -1
  return 1

@api_op
def rm(filename):
  try:
    if os.path.exists(filename):
      os.unlink(filename)
    return True
  except:
    traceback.print_exc()
    return False

import contextlib

@api_op
@contextlib.contextmanager
def with_graph(graph):
  prev = getattr(state, 'graph', None)
  if graph == prev:
    yield
    return
  graph._unsafe_unfinalize()
  state.graph = graph
  try:
    with graph.as_default():
      yield
  finally:
    state.graph = prev
    graph.finalize()

@api_op
@contextlib.contextmanager
def with_session(session):
  prev = getattr(state, 'session', None)
  if session == prev:
    yield
    return
  state.session = session
  try:
    with session.as_default():
      with api.with_graph(session.graph):
        yield
  finally:
    state.session = prev

@api_op
def unique_variables(var_list=None):
  r = []
  if var_list is None or var_list == 'all':
    var_list = tf.global_variables() + tf.local_variables() + tf.model_variables() + tf.trainable_variables()
  elif isinstance(var_list, str):
    names = var_list.split()
    var_list = []
    for name in names:
      name = name.strip().lower()
      if name == 'globals' or name == 'global':
        var_list += tf.global_variables()
      elif name == 'locals' or name == 'local':
        var_list += tf.local_variables()
      elif name == 'models' or name == 'model':
        var_list += tf.model_variables()
      elif name == 'trainable' or name == 'trainables' or name == 'trains' or name == 'train':
        var_list += tf.trainable_variables()
      else:
        var_list += tf.get_collection(name)
  assert isinstance(var_list, list)
  for v in var_list:
    if v not in r:
      r.append(v)
  return r

@api_op
def rollback(ckpt, step=None, global_step=None, session=None, var_list=None):
  logging.info('ttf.rollback(ckpt=%r, step=%r, var_list=%r)', ckpt, step, var_list)
  session = session or tf.get_default_session()
  if global_step is None:
    global_step = tf.train.get_global_step()
  assert global_step is not None
  if step is None:
    step = api.eval_lightweight(global_step, session=session)
  var_list = api.unique_variables(var_list)
  saver = tf.train.Saver(var_list=var_list)
  saver.restore(session, ckpt)
  api.load_lightweight(global_step, step)
  return saver

@api_op
def break_session(session=None):
  logging.info('Debug breakpoint...')
  if session is None:
    logging.warning('Session is None')
    import pdb
    pdb.set_trace()
  else:
    with api.with_session(session):
      import pdb
      pdb.set_trace()

@api_op
def restore(step, session=None, override_step=None):
  session = session or tf.get_default_session()
  ckpt = os.path.join(os.environ['MODEL_DIR'], 'model.ckpt-{}'.format(step))
  with api.with_session(session):
    with api.with_graph(session.graph):
      api.rollback(ckpt, var_list=api.unique_variables('trains') + vals.gan._disc_optimizer_opt.variables() + vals.gan._gen_optimizer_opt.variables())
      # TODO: figure out how to roll back the hook timers.
      # all_vars = unique_variables()
      # step_vars = [x for x in all_vars if api.variable_name(x).startswith('global_step')]
      # set_step = override_step if override_step is not None else step
      # for v in step_vars:
      #   logging.info('Setting %s to %s', v, set_step)
      #   load_lightweight(v, set_step)

@api_op
def heartbeat(session=None):
  logging.info('Heartbeat')
  session = session or tf.get_default_session()
  if session is not None and session not in state.pinned_sessions:
    state.pinned_sessions.append(session)
  if os.path.exists('debug_break.txt'):
    api.rm('debug_break.txt')
    api.break_session(session=session)
  try:
    if os.path.exists('restore_step.txt'):
      with open('restore_step.txt') as f:
        chars = f.read()
      api.rm('restore_step.txt')
      parts = chars.strip().split()
      if len(parts) == 1:
        api.restore(int(parts[0]), session=session)
      elif len(parts) == 2:
        api.restore(int(parts[0]), session=session, override_step=int(parts[1]))
      else:
        logging.warn("Invalid restore_step %r", chars)
  except:
    traceback.print_exc()

@api_op
def update_vars(name=None, skip_unknown=False, session=None):
  session = session or tf.get_default_session()
  if session is None:
    logging.warning('Session is None')
    return
  tensorfork.reload(name=name, skip_unknown=skip_unknown)
  for entry in state.vars.values():
    knob = entry['knob']
    variable = entry['variable']
    vm_value = tensorfork.knobs(knob)
    tf_value = api.eval_lightweight(variable, session=session)
    logging.info('Setting knob %s to %s (was %s)', knob, vm_value, tf_value)
    api.load_lightweight(variable, vm_value, session=session)
  logging.info('Session %s cwd %s', session, os.getcwd())

if __name__ == "__main__":
  logging.set_verbosity(0)
  tensorfork.tensorfork.tf = importlib.__import__('tensorfork_tf')
  #sess = tf1.InteractiveSession()
  tensorfork.main()

