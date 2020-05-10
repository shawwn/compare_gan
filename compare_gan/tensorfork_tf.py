from . import tensorfork
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import importlib
import threading
import gin
import gin.tf.external_configurables

class Context:
  def __init__(self):
    pass

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

def eval_lightweight(variable, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return None
  if timeout_in_ms is None:
    timeout_in_ms = state.eval_lightweight_timeout
  return eval(variable, session=session, timeout_in_ms=state.eval_lightweight_timeout)

def load_lightweight(variable, value, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return
  if timeout_in_ms is None:
    timeout_in_ms = state.load_lightweight_timeout
  return load(variable, value, session=session, timeout_in_ms=timeout_in_ms)

from tensorflow.core.protobuf import config_pb2

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

def eval(variable, session=None, timeout_in_ms=None):
  session = session or tf.get_default_session()
  if session is None:
    return None
  options = None
  if timeout_in_ms:
    options=config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
  return session.run(variable, options=options)

import re

def variable_name(variable):
  if hasattr(variable, 'name'):
    variable = variable.name
  if re.match(r'core[0-9]+/', variable):
    variable = variable.split('/', 1)[-1]
  variable = variable.split(':', 1)[0]
  return variable

def grab_values(variables, reader, reshape=False):
  for variable in variables:
    name = variable_name(variable).split(':')[0]
    value = reader.get_tensor(name)
    value = truncate_value(variable, value, reshape=reshape)
    yield variable, value

import numpy as np
import math
import sys

def truncate_value(variable, value, reshape=True):
  if not reshape:
    return value
  shape = variable.shape.as_list()
  params = np.prod(shape)
  params2 = np.prod(value.shape)
  if params == params2:
    return value
  if params2 > params:
    logging.info('Truncating {} from shape {} to shape {}. var={}', variable.name, value.shape, shape, variable)
    sys.stdout.flush()
    value = np.array(value)
    value = value.reshape([-1])
    value = value[0:params]
    value = value.reshape(shape)
  else:
    logging.info('Expanding {} from shape {} to shape {}. var={}', variable.name, value.shape, shape, variable)
    sys.stdout.flush()
    value = np.array(value)
    value = value.reshape([-1])
    n = math.ceil(params / params2)
    value = np.tile(value, n)
    value = value.reshape(shape)
  return value

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

def fqn(name, scope=None):
  if scope is None or len(scope) <= 0:
    scope = tf.get_variable_scope().name
  name = '_'.join(re.split('[^a-z0-9]', os.path.join(scope, name)))
  return name

def absolute_name_scope(scope):
  return tf.name_scope(scope + "/")

def absolute_variable_scope(scope=None, **kwargs):
  if scope is None:
    scope = tf.get_variable_scope().name
  return tf.variable_scope(tf.VariableScope(name=scope, **kwargs), auxiliary_name_scope=False)

def scoped_name(name=None, scope=None):
  if scope is None:
    scope = tf.get_variable_scope().name
  if name is None:
    name = ''
  name = name.split(':')[0]
  return os.path.join(scope, name)

def get_var(name, default_value=None, update=False, scope=None, dtype=None, shape=(), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES], use_resource=True, **kws):
  if scope is not None:
    try:
      return get_var(os.path.join(scope, name), default_value=None, update=update, scope=None, dtype=dtype, shape=shape, trainable=trainable, collections=collections, use_resource=use_resource, **kws)
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
    load_lightweight(var, value)
  v = eval_lightweight(var)
  if v is not None and v != value:
    tensorfork.knobs(name, v, update=True)
  return var

from absl import logging

def compare_values(a, b, eps=1e-10):
  if abs(a - b) < eps:
    return 0
  if a > b:
    return -1
  return 1

def rm(filename):
  try:
    if os.path.exists(filename):
      os.unlink(filename)
    return True
  except:
    import traceback
    traceback.print_exc()
    return False

def update_vars(name=None, skip_unknown=False, session=None):
  session = session or tf.get_default_session()
  if session is None:
    logging.warning('Session is None')
    return
  if session not in state.pinned_sessions:
    state.pinned_sessions.append(session)
  state.session = session
  try:
    tensorfork.reload(name=name, skip_unknown=skip_unknown)
  finally:
    if os.path.exists('debug_break.txt'):
      logging.info('Debug breakpoint...')
      rm('debug_break.txt')
      import pdb
      pdb.set_trace()
  for entry in state.vars.values():
    knob = entry['knob']
    variable = entry['variable']
    vm_value = tensorfork.knobs(knob)
    tf_value = eval_lightweight(variable, session=session)
    logging.info('Setting knob %s to %s (was %s)', knob, vm_value, tf_value)
    load_lightweight(variable, vm_value, session=session)
    logging.info('Session %s cwd %s', session, os.getcwd())

if __name__ == "__main__":
  logging.set_verbosity(0)
  tensorfork.tensorfork.tf = importlib.__import__('tensorfork_tf')
  #sess = tf1.InteractiveSession()
  tensorfork.main()

